# Python Notes Library

A small, dependency-free Python library for managing notes in memory.

## Quickstart

No external dependencies are required. This library uses only the Python standard library.

python
from notes import Note, NotesStore

# Create a store
store = NotesStore()

# Add notes
note1 = store.add(Note(title="Buy milk", body="Skim milk please"))
note2 = store.add(Note(title="Write report", body="Quarterly report"))

# List all notes
for note in store.list_all():
    print(note)

# Mark a note as done
store.mark_done(note1)

# Delete a note
store.delete(note2)


## API

### Note

A dataclass with three fields:
- `title` (str): The note title
- `body` (str): The note content
- `done` (bool): Whether the note is completed (default: False)

### NotesStore

An in-memory store for Note objects, keyed by integer IDs.

- `add(note: Note) -> int`: Add a note and return its assigned ID
- `list_all() -> list[Note]`: Return all stored notes in insertion order
- `mark_done(note_id: int) -> None`: Mark a note as done by its ID
- `delete(note_id: int) -> None`: Remove a note by its ID

Both `mark_done` and `delete` raise `ValueError` if the note ID is not found.
