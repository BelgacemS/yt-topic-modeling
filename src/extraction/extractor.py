"""
Module d'extraction des commentaires YouTube.
Utilise yt-dlp pour récupérer commentaires et métadonnées des vidéos.
"""

import re
import json
import argparse
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp


class CommentExtractor:
    """Classe principale pour extraire les commentaires YouTube."""

    def __init__(self, output_dir="data/raw", max_workers=4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

        # fichier de progression pour reprendre si on est interrompu
        self.progress_file = self.output_dir / ".progress.json"
        self.progress = self._load_progress()
        self._lock = threading.Lock()

    def _load_progress(self):
        """Charge le fichier de progression si il existe."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Fichier de progression corrompu, on repart de zéro")
                return {"done": [], "failed": {}}
        return {"done": [], "failed": {}}

    def _save_progress(self):
        # thread-safe car appelé depuis les workers
        with self._lock:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2, ensure_ascii=False)

    def _is_done(self, video_id):
        return video_id in self.progress["done"]

    def _mark_done(self, video_id):
        with self._lock:
            if video_id not in self.progress["done"]:
                self.progress["done"].append(video_id)
            self.progress["failed"].pop(video_id, None)
        self._save_progress()

    def _mark_failed(self, video_id, reason):
        with self._lock:
            self.progress["failed"][video_id] = reason
        self._save_progress()

    # ---- récupération des vidéos d'une chaîne ----

    def get_channel_videos(self, channel_url, max_videos=None):
        """Récupère la liste des IDs de vidéos d'une chaîne YouTube."""
        print(f"\nRécupération des vidéos de {channel_url}...")

        opts = {
            'quiet': True,
            'extract_flat': True,
            'no_warnings': True,
        }
        if max_videos:
            opts['playlistend'] = max_videos

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)

                if info is None:
                    print(f"Impossible de récupérer {channel_url}")
                    return []

                if 'entries' in info:
                    videos = []
                    for entry in info['entries']:
                        if entry is None:
                            continue
                        vid_id = entry.get('id')
                        if vid_id:
                            videos.append(vid_id)
                    print(f"  -> {len(videos)} vidéos trouvées")
                    return videos
                else:
                    # c'est juste une vidéo, pas une chaîne
                    return [info['id']] if info.get('id') else []

        except Exception as e:
            print(f"Erreur pour {channel_url}: {e}")
            return []

    # ---- extraction d'une vidéo ----

    def extract_video(self, video_id):
        """Extrait les commentaires d'une vidéo YouTube avec yt-dlp."""

        if self._is_done(video_id):
            print(f"  [{video_id}] déjà extrait, skip")
            return None

        # si le fichier json existe déjà on le considère comme fait
        output_file = self.output_dir / f"{video_id}.json"
        if output_file.exists():
            print(f"  [{video_id}] fichier existant, skip")
            self._mark_done(video_id)
            return None

        url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"  [{video_id}] extraction en cours...")

        opts = {
            'quiet': True,
            'no_warnings': True,
            'getcomments': True,
            'skip_download': True,
            # TODO: ajouter un paramètre pour limiter le nombre de commentaires
            'extractor_args': {
                'youtube': {
                    'max_comments': ['all', 'all', 'all', 'all'],
                },
            },
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)

            if info is None:
                self._mark_failed(video_id, "yt-dlp a retourné None")
                return None

            data = self._format_video_data(info)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            nb = len(data['comments'])
            print(f"  [{video_id}] OK - {nb} commentaires")
            self._mark_done(video_id)
            return data

        except yt_dlp.utils.DownloadError as e:
            err_msg = str(e)

            # commentaires désactivés -> on essaie de sauver les métadonnées quand même
            if "comments" in err_msg.lower() or "disabled" in err_msg.lower():
                print(f"  [{video_id}] commentaires désactivés, sauvegarde sans commentaires")
                return self._save_without_comments(video_id)
            else:
                print(f"  [{video_id}] erreur yt-dlp: {err_msg[:100]}")
                self._mark_failed(video_id, err_msg[:200])
                return None

        except Exception as e:
            print(f"  [{video_id}] erreur inattendue: {e}")
            self._mark_failed(video_id, str(e)[:200])
            return None

    def _save_without_comments(self, video_id):
        # commentaires désactivés → on tente de sauver au moins les métadonnées
        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            opts = {'quiet': True, 'no_warnings': True, 'skip_download': True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)

            if info is None:
                self._mark_failed(video_id, "impossible de récupérer les métadonnées")
                return None

            data = self._format_video_data(info)
            output_file = self.output_dir / f"{video_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"  [{video_id}] sauvegardé (0 commentaires)")
            self._mark_done(video_id)
            return data

        except Exception as e:
            print(f"  [{video_id}] impossible de récupérer les métadonnées: {e}")
            self._mark_failed(video_id, f"no comments + metadata failed: {e}")
            return None

    # ---- formatage des données ----

    def _format_video_data(self, info):
        # date d'upload : yt-dlp donne YYYYMMDD, nous on veut YYYY-MM-DD
        upload_date = info.get('upload_date', '')
        if upload_date and len(upload_date) == 8:
            upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"

        raw_comments = info.get('comments') or []
        comments = self._build_comment_tree(raw_comments)

        return {
            "video_id": info.get('id', ''),
            "title": info.get('title', ''),
            "channel": info.get('channel', info.get('uploader', '')),
            "upload_date": upload_date,
            "comments": comments,
        }

    def _build_comment_tree(self, raw_comments):
        """Organise les commentaires en arbre : top-level avec leurs replies."""

        top_level = {}
        replies_map = {}

        for c in raw_comments:
            formatted = self._format_comment(c)
            parent = c.get('parent', 'root')

            if parent == 'root':
                top_level[formatted['id']] = formatted
            else:
                if parent not in replies_map:
                    replies_map[parent] = []
                replies_map[parent].append(formatted)

        # on rattache les replies à leur commentaire parent
        result = []
        for cid, comment in top_level.items():
            comment['replies'] = replies_map.get(cid, [])
            result.append(comment)

        return result

    def _format_comment(self, c):
        ts = c.get('timestamp')
        if ts:
            try:
                ts = datetime.fromtimestamp(ts).isoformat()
            except (OSError, ValueError):
                ts = None

        return {
            "id": c.get('id', ''),
            "text": c.get('text', ''),
            "author": c.get('author', ''),
            "timestamp": ts,
            "likes": c.get('like_count', 0) or 0,
        }

    # ---- orchestration ----

    def run(self, channels=None, videos=None, max_videos=None):
        """Point d'entrée principal. Extrait les commentaires des chaînes/vidéos données."""

        all_ids = []

        # récupérer les vidéos depuis les chaînes
        if channels:
            for ch in channels:
                ch = ch.strip()
                if not ch or ch.startswith('#'):
                    continue
                vids = self.get_channel_videos(ch, max_videos=max_videos)
                all_ids.extend(vids)

        # ajouter les vidéos individuelles
        if videos:
            for v in videos:
                v = v.strip()
                if not v:
                    continue
                vid_id = parse_video_id(v)
                if vid_id:
                    all_ids.append(vid_id)

        # dédupliquer en gardant l'ordre
        all_ids = list(dict.fromkeys(all_ids))

        # filtrer ceux déjà faits
        todo = [v for v in all_ids if not self._is_done(v)]

        print(f"\n{'='*50}")
        print(f"Total: {len(all_ids)} vidéos, {len(todo)} à traiter")
        print(f"{'='*50}\n")

        if not todo:
            print("Rien à faire !")
            return

        # extraction (parallèle ou séquentielle)
        if self.max_workers > 1 and len(todo) > 1:
            self._extract_parallel(todo)
        else:
            for vid_id in todo:
                self.extract_video(vid_id)

        # petit résumé à la fin
        nb_done = len(self.progress["done"])
        nb_failed = len(self.progress["failed"])
        print(f"\n{'='*50}")
        print(f"Terminé ! {nb_done} extraits, {nb_failed} en erreur")
        if nb_failed > 0:
            print(f"Vidéos échouées: {list(self.progress['failed'].keys())}")
        print(f"{'='*50}")

    def _extract_parallel(self, video_ids):
        """Lance l'extraction en parallèle avec ThreadPoolExecutor."""
        print(f"Lancement avec {self.max_workers} workers...\n")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.extract_video, vid): vid
                for vid in video_ids
            }

            for future in as_completed(futures):
                vid_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    # normalement géré dans extract_video, mais au cas où
                    print(f"  [{vid_id}] erreur thread: {e}")
                    self._mark_failed(vid_id, str(e))


# ---- utilitaires ----

def parse_video_id(url_or_id):
    """Extrait l'ID YouTube depuis une URL ou retourne l'ID directement."""
    url_or_id = url_or_id.strip()

    # déjà un ID ? (11 chars alphanumériques + tirets/underscores)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
        return url_or_id

    # sinon on essaie de parser l'URL
    patterns = [
        r'(?:v=|/v/)([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
        r'shorts/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    # en dernier recours on retourne tel quel, yt-dlp se débrouillera
    return url_or_id


def main():
    parser = argparse.ArgumentParser(
        description="Extraction de commentaires YouTube via yt-dlp"
    )
    parser.add_argument(
        '--channels', type=str,
        help="Fichier texte avec les URLs des chaînes (une par ligne)"
    )
    parser.add_argument(
        '--videos', type=str, nargs='+',
        help="URLs ou IDs de vidéos individuelles"
    )
    parser.add_argument(
        '--output', type=str, default='data/raw',
        help="Dossier de sortie (défaut: data/raw)"
    )
    parser.add_argument(
        '--max-videos', type=int, default=None,
        help="Nombre max de vidéos par chaîne"
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help="Nombre de threads (défaut: 4)"
    )

    args = parser.parse_args()

    if not args.channels and not args.videos:
        parser.error("Il faut --channels ou --videos (ou les deux)")

    # lire le fichier de chaînes
    channels = None
    if args.channels:
        p = Path(args.channels)
        if not p.exists():
            print(f"Erreur: fichier '{args.channels}' introuvable")
            return
        channels = p.read_text().strip().splitlines()

    extractor = CommentExtractor(
        output_dir=args.output,
        max_workers=args.workers,
    )
    extractor.run(
        channels=channels,
        videos=args.videos,
        max_videos=args.max_videos,
    )


if __name__ == '__main__':
    main()
