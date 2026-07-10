# Python Notes Library

A small, dependency-free Python library for managing notes in memory.

## Installation

No installation needed — uses only the Python standard library.

## Usage

python
from notes import NotesStore

# Create a store
store = NotesStore()

# Add a note
note_id = store.add("Meeting Notes", "Discuss Q3 goals")
print(f"Added note with id: {note_id}")

# List all notes
for note in store.list_all():
    print(f"[{note.id}] {note.title} - done={note.done}")

# Mark a note as done
store.mark_done(note_id)

# Delete a note
store.delete(note_id)


## API

### `Note` dataclass

| Field  | Type   | Default |
|--------|--------|---------|
| title  | str    | (required) |
| body   | str    | (required) |
| done   | bool   | False   |
| id     | str    | auto-generated uuid4 |

### `NotesStore`

- `add(title: str, body: str) -> str` — Adds a note, returns its ID.
- `list_all() -> List[Note]` — Returns all notes.
- `mark_done(note_id: str) -> None` — Marks a note as done. Raises `ValueError` if the ID is not found.
- `delete(note_id: str) -> None` — Deletes a note. Raises `ValueError` if the ID is not found.

## Running Tests

bash
pip install pytest
pytest test_notes.py -v

