import pytest

from notes import Note, NotesStore


class TestNoteDataclass:
    def test_note_defaults(self):
        note = Note(title="Hello", body="World")
        assert note.title == "Hello"
        assert note.body == "World"
        assert note.done is False
        assert note.id is not None

    def test_note_explicit_done(self):
        note = Note(title="T", body="B", done=True)
        assert note.done is True

    def test_note_unique_ids(self):
        n1 = Note(title="A", body="B")
        n2 = Note(title="C", body="D")
        assert n1.id != n2.id


class TestNotesStoreAdd:
    def test_add_returns_id(self):
        store = NotesStore()
        note_id = store.add("Title", "Body")
        assert isinstance(note_id, str)
        assert len(note_id) > 0

    def test_add_stores_note(self):
        store = NotesStore()
        note_id = store.add("Title", "Body")
        notes = store.list_all()
        assert len(notes) == 1
        assert notes[0].id == note_id
        assert notes[0].title == "Title"
        assert notes[0].body == "Body"
        assert notes[0].done is False

    def test_add_multiple(self):
        store = NotesStore()
        id1 = store.add("First", "Body 1")
        id2 = store.add("Second", "Body 2")
        assert id1 != id2
        assert len(store.list_all()) == 2


class TestNotesStoreListAll:
    def test_list_all_empty(self):
        store = NotesStore()
        assert store.list_all() == []

    def test_list_all_returns_copies(self):
        store = NotesStore()
        store.add("A", "B")
        notes = store.list_all()
        assert len(notes) == 1
        # Modifying the returned list shouldn't affect the store
        notes.clear()
        assert len(store.list_all()) == 1


class TestNotesStoreMarkDone:
    def test_mark_done(self):
        store = NotesStore()
        note_id = store.add("Title", "Body")
        store.mark_done(note_id)
        notes = store.list_all()
        assert notes[0].done is True

    def test_mark_done_missing_id(self):
        store = NotesStore()
        with pytest.raises(ValueError, match="not found"):
            store.mark_done("nonexistent-id")

    def test_mark_done_preserves_other_notes(self):
        store = NotesStore()
        id1 = store.add("A", "B")
        id2 = store.add("C", "D")
        store.mark_done(id1)
        notes = store.list_all()
        assert notes[0].done is True
        assert notes[1].done is False


class TestNotesStoreDelete:
    def test_delete(self):
        store = NotesStore()
        note_id = store.add("Title", "Body")
        store.delete(note_id)
        assert len(store.list_all()) == 0

    def test_delete_missing_id(self):
        store = NotesStore()
        with pytest.raises(ValueError, match="not found"):
            store.delete("nonexistent-id")

    def test_delete_preserves_other_notes(self):
        store = NotesStore()
        id1 = store.add("A", "B")
        id2 = store.add("C", "D")
        store.delete(id1)
        notes = store.list_all()
        assert len(notes) == 1
        assert notes[0].id == id2
