"""Tests pour le module d'extraction des commentaires YouTube."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.extraction.extractor import CommentExtractor, parse_video_id


# ---- fixtures ----

@pytest.fixture
def extractor(tmp_path):
    """Crée un extracteur avec un dossier temporaire."""
    return CommentExtractor(output_dir=str(tmp_path), max_workers=1)


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path


# ---- données de test ----

FAKE_VIDEO_INFO = {
    'id': 'dQw4w9WgXcQ',
    'title': 'Rick Astley - Never Gonna Give You Up',
    'channel': 'Rick Astley',
    'uploader': 'Rick Astley',
    'upload_date': '20091025',
    'comments': [
        {
            'id': 'comment1',
            'text': 'Super vidéo !',
            'author': 'Alice',
            'timestamp': 1705401000,
            'like_count': 42,
            'parent': 'root',
        },
        {
            'id': 'comment2',
            'text': 'Tellement classique',
            'author': 'Bob',
            'timestamp': 1705402000,
            'like_count': 10,
            'parent': 'root',
        },
        {
            'id': 'reply1',
            'text': 'Carrément !',
            'author': 'Charlie',
            'timestamp': 1705403000,
            'like_count': 3,
            'parent': 'comment1',
        },
        {
            'id': 'reply2',
            'text': 'Trop bien',
            'author': 'Diana',
            'timestamp': 1705404000,
            'like_count': 1,
            'parent': 'comment1',
        },
    ],
}

FAKE_VIDEO_NO_COMMENTS = {
    'id': 'nocomm12345',
    'title': 'Vidéo sans commentaires',
    'channel': 'Test Channel',
    'upload_date': '20240301',
    'comments': [],
}

FAKE_CHANNEL_INFO = {
    '_type': 'playlist',
    'id': 'UCtest',
    'title': 'Ma Chaîne Test',
    'entries': [
        {'id': 'vid001', 'title': 'Vidéo 1', 'url': 'https://youtube.com/watch?v=vid001'},
        {'id': 'vid002', 'title': 'Vidéo 2', 'url': 'https://youtube.com/watch?v=vid002'},
        {'id': 'vid003', 'title': 'Vidéo 3', 'url': 'https://youtube.com/watch?v=vid003'},
        None,  # parfois yt-dlp retourne des None dans les entries
    ],
}


# ---- tests du parsing d'URL ----

class TestParseVideoId:

    def test_id_direct(self):
        assert parse_video_id('dQw4w9WgXcQ') == 'dQw4w9WgXcQ'

    def test_url_standard(self):
        assert parse_video_id('https://www.youtube.com/watch?v=dQw4w9WgXcQ') == 'dQw4w9WgXcQ'

    def test_url_courte(self):
        assert parse_video_id('https://youtu.be/dQw4w9WgXcQ') == 'dQw4w9WgXcQ'

    def test_url_shorts(self):
        assert parse_video_id('https://www.youtube.com/shorts/dQw4w9WgXcQ') == 'dQw4w9WgXcQ'

    def test_url_avec_params(self):
        res = parse_video_id('https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42')
        assert res == 'dQw4w9WgXcQ'

    def test_espaces(self):
        assert parse_video_id('  dQw4w9WgXcQ  ') == 'dQw4w9WgXcQ'

    def test_url_embed(self):
        assert parse_video_id('https://www.youtube.com/v/dQw4w9WgXcQ') == 'dQw4w9WgXcQ'


# ---- tests du formatage ----

class TestFormatComment:

    def test_format_basique(self, extractor):
        raw = {
            'id': 'c123',
            'text': 'Hello world',
            'author': 'User1',
            'timestamp': 1705401000,
            'like_count': 5,
            'parent': 'root',
        }
        res = extractor._format_comment(raw)

        assert res['id'] == 'c123'
        assert res['text'] == 'Hello world'
        assert res['author'] == 'User1'
        assert res['likes'] == 5
        # le timestamp doit être en ISO format
        assert 'T' in res['timestamp']

    def test_comment_sans_likes(self, extractor):
        raw = {
            'id': 'c456',
            'text': 'test',
            'author': 'User2',
            'timestamp': None,
            'like_count': None,
            'parent': 'root',
        }
        res = extractor._format_comment(raw)
        assert res['likes'] == 0
        assert res['timestamp'] is None

    def test_comment_champs_manquants(self, extractor):
        """Un commentaire avec des champs manquants ne doit pas crasher."""
        res = extractor._format_comment({})
        assert res['id'] == ''
        assert res['text'] == ''
        assert res['likes'] == 0


class TestBuildCommentTree:

    def test_arbre_basique(self, extractor):
        raw = FAKE_VIDEO_INFO['comments']
        tree = extractor._build_comment_tree(raw)

        # 2 commentaires top-level
        assert len(tree) == 2

        # le premier commentaire a 2 replies
        c1 = next(c for c in tree if c['id'] == 'comment1')
        assert len(c1['replies']) == 2
        assert c1['replies'][0]['id'] == 'reply1'
        assert c1['replies'][1]['id'] == 'reply2'

        # le deuxième n'a pas de replies
        c2 = next(c for c in tree if c['id'] == 'comment2')
        assert len(c2['replies']) == 0

    def test_liste_vide(self, extractor):
        assert extractor._build_comment_tree([]) == []


class TestFormatVideoData:

    def test_format_complet(self, extractor):
        data = extractor._format_video_data(FAKE_VIDEO_INFO)

        assert data['video_id'] == 'dQw4w9WgXcQ'
        assert data['title'] == 'Rick Astley - Never Gonna Give You Up'
        assert data['channel'] == 'Rick Astley'
        # la date doit être reformatée
        assert data['upload_date'] == '2009-10-25'
        assert len(data['comments']) == 2

    def test_format_sans_commentaires(self, extractor):
        data = extractor._format_video_data(FAKE_VIDEO_NO_COMMENTS)
        assert data['video_id'] == 'nocomm12345'
        assert data['comments'] == []

    def test_upload_date_vide(self, extractor):
        info = {**FAKE_VIDEO_INFO, 'upload_date': ''}
        data = extractor._format_video_data(info)
        assert data['upload_date'] == ''

    def test_fallback_uploader(self, extractor):
        """Si 'channel' est absent, on utilise 'uploader'."""
        info = dict(FAKE_VIDEO_INFO)
        del info['channel']
        data = extractor._format_video_data(info)
        assert data['channel'] == 'Rick Astley'


# ---- tests du progrès ----

class TestProgress:

    def test_progress_initial(self, extractor):
        assert extractor.progress == {"done": [], "failed": {}}

    def test_mark_done(self, extractor):
        extractor._mark_done('vid001')
        assert 'vid001' in extractor.progress['done']
        # vérifier que c'est persisté sur le disque
        assert extractor.progress_file.exists()
        saved = json.loads(extractor.progress_file.read_text())
        assert 'vid001' in saved['done']

    def test_mark_done_dedup(self, extractor):
        """Marquer done deux fois ne doit pas dupliquer."""
        extractor._mark_done('vid001')
        extractor._mark_done('vid001')
        assert extractor.progress['done'].count('vid001') == 1

    def test_mark_failed(self, extractor):
        extractor._mark_failed('vid002', 'timeout')
        assert extractor.progress['failed']['vid002'] == 'timeout'

    def test_mark_done_enleve_failed(self, extractor):
        """Si une vidéo fail puis réussit, elle sort des failed."""
        extractor._mark_failed('vid001', 'erreur')
        assert 'vid001' in extractor.progress['failed']
        extractor._mark_done('vid001')
        assert 'vid001' not in extractor.progress['failed']
        assert 'vid001' in extractor.progress['done']

    def test_reload_progress(self, tmp_output):
        """Le progrès doit survivre à la recréation de l'extracteur."""
        ext1 = CommentExtractor(output_dir=str(tmp_output), max_workers=1)
        ext1._mark_done('vid001')
        ext1._mark_done('vid002')
        ext1._mark_failed('vid003', 'rate limit')

        # on recrée un extracteur sur le même dossier
        ext2 = CommentExtractor(output_dir=str(tmp_output), max_workers=1)
        assert 'vid001' in ext2.progress['done']
        assert 'vid002' in ext2.progress['done']
        assert ext2.progress['failed']['vid003'] == 'rate limit'

    def test_progress_corrompu(self, tmp_output):
        """Un fichier de progression corrompu ne doit pas crasher."""
        progress_file = tmp_output / ".progress.json"
        progress_file.write_text("{invalid json")

        ext = CommentExtractor(output_dir=str(tmp_output), max_workers=1)
        assert ext.progress == {"done": [], "failed": {}}


# ---- tests d'extraction (mockés) ----

class TestExtractVideo:

    @patch('src.extraction.extractor.yt_dlp.YoutubeDL')
    def test_extraction_ok(self, mock_ydl_class, extractor, tmp_output):
        """Test d'une extraction qui se passe bien."""
        # on mock le context manager de YoutubeDL
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = FAKE_VIDEO_INFO

        result = extractor.extract_video('dQw4w9WgXcQ')

        assert result is not None
        assert result['video_id'] == 'dQw4w9WgXcQ'
        assert len(result['comments']) == 2

        # vérifier que le fichier json est créé
        output_file = Path(extractor.output_dir) / 'dQw4w9WgXcQ.json'
        assert output_file.exists()

        # vérifier le contenu du fichier
        saved = json.loads(output_file.read_text())
        assert saved['video_id'] == 'dQw4w9WgXcQ'
        assert saved['title'] == 'Rick Astley - Never Gonna Give You Up'

        # vérifier le progrès
        assert 'dQw4w9WgXcQ' in extractor.progress['done']

    def test_skip_deja_fait(self, extractor):
        """Une vidéo déjà dans le progrès est skip."""
        extractor._mark_done('already_done')
        result = extractor.extract_video('already_done')
        assert result is None

    def test_skip_fichier_existant(self, extractor, tmp_output):
        """Si le fichier json existe déjà, on skip."""
        # créer un fichier json bidon
        (tmp_output / 'existing123.json').write_text('{}')
        result = extractor.extract_video('existing123')
        assert result is None
        assert 'existing123' in extractor.progress['done']

    @patch('src.extraction.extractor.yt_dlp.YoutubeDL')
    def test_extraction_yt_dlp_none(self, mock_ydl_class, extractor):
        """Si yt-dlp retourne None, on marque comme failed."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = None

        result = extractor.extract_video('null_video')
        assert result is None
        assert 'null_video' in extractor.progress['failed']

    @patch('src.extraction.extractor.yt_dlp.YoutubeDL')
    def test_extraction_erreur(self, mock_ydl_class, extractor):
        """Test d'une erreur yt-dlp quelconque."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.side_effect = Exception("Network error")

        result = extractor.extract_video('error_video')
        assert result is None
        assert 'error_video' in extractor.progress['failed']


# ---- tests récupération de chaîne ----

class TestGetChannelVideos:

    @patch('src.extraction.extractor.yt_dlp.YoutubeDL')
    def test_channel_ok(self, mock_ydl_class, extractor):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = FAKE_CHANNEL_INFO

        videos = extractor.get_channel_videos('https://youtube.com/@test')

        assert len(videos) == 3
        assert 'vid001' in videos
        assert 'vid002' in videos
        assert 'vid003' in videos

    @patch('src.extraction.extractor.yt_dlp.YoutubeDL')
    def test_channel_none(self, mock_ydl_class, extractor):
        """Si yt-dlp retourne None pour une chaîne."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = None

        videos = extractor.get_channel_videos('https://youtube.com/@nonexistent')
        assert videos == []

    @patch('src.extraction.extractor.yt_dlp.YoutubeDL')
    def test_channel_erreur(self, mock_ydl_class, extractor):
        """Si yt-dlp lève une exception."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.side_effect = Exception("403 Forbidden")

        videos = extractor.get_channel_videos('https://youtube.com/@test')
        assert videos == []


# ---- test du run complet ----

class TestRun:

    @patch.object(CommentExtractor, 'extract_video')
    @patch.object(CommentExtractor, 'get_channel_videos')
    def test_run_avec_channels(self, mock_get_vids, mock_extract, extractor):
        mock_get_vids.return_value = ['vid001', 'vid002']
        mock_extract.return_value = None

        extractor.run(channels=['https://youtube.com/@test'])

        assert mock_get_vids.called
        assert mock_extract.call_count == 2

    @patch.object(CommentExtractor, 'extract_video')
    def test_run_avec_videos(self, mock_extract, extractor):
        mock_extract.return_value = None

        extractor.run(videos=['dQw4w9WgXcQ', 'abc12345678'])

        assert mock_extract.call_count == 2

    @patch.object(CommentExtractor, 'extract_video')
    def test_run_skip_done(self, mock_extract, extractor):
        """Les vidéos déjà faites ne sont pas re-traitées."""
        extractor._mark_done('vid001')
        mock_extract.return_value = None

        extractor.run(videos=['vid001', 'vid002'])

        # seulement vid002 doit être extrait
        assert mock_extract.call_count == 1
        mock_extract.assert_called_once_with('vid002')

    @patch.object(CommentExtractor, 'extract_video')
    def test_run_dedup(self, mock_extract, extractor):
        """Les doublons sont éliminés."""
        mock_extract.return_value = None

        extractor.run(videos=['vid001', 'vid001', 'vid001'])
        assert mock_extract.call_count == 1

    def test_run_rien(self, extractor):
        """Si tout est déjà fait, run ne crashe pas."""
        extractor._mark_done('vid001')
        extractor.run(videos=['vid001'])


# ---- test du CLI ----

class TestCLI:

    def test_channels_file(self, tmp_path):
        """Test que le fichier channels est bien lu."""
        channels_file = tmp_path / "channels.txt"
        channels_file.write_text("https://youtube.com/@test1\nhttps://youtube.com/@test2\n# commentaire\n")

        lines = channels_file.read_text().strip().splitlines()
        # filtrer les commentaires comme le fait run()
        filtered = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]
        assert len(filtered) == 2


# ---- test du parallélisme ----

class TestParallel:

    @patch.object(CommentExtractor, 'extract_video')
    def test_parallel_extraction(self, mock_extract, tmp_output):
        """Test que l'extraction parallèle fonctionne."""
        mock_extract.return_value = None
        extractor = CommentExtractor(output_dir=str(tmp_output), max_workers=2)

        extractor.run(videos=['vid001', 'vid002', 'vid003'])
        assert mock_extract.call_count == 3
