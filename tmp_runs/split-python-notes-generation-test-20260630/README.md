# Python Notes Library

A small, dependency-free Python library for managing notes in memory.

## Features

- `Note` dataclass with title, body, and done status
- `NotesStore` for in-memory CRUD operations
- Input validation (rejects empty titles/bodies)
- Unique IDs generated via `uuid4`

## Usage

python
from notes import Note, NotesStore

# Initialize the store
store = NotesStore()

# Add a note
note = store.add(title="Meeting Notes", body="Discussed Q3 roadmap")
print(note.id)  # e.g., 'a1b2c3d4-...'

# List all notes
all_notes = store.list_all()
for n in all_notes:
    print(f"[{n.done}] {n.title}: {n.body}")

# Mark a note as done
store.mark_done(note.id)

# Delete a note
store.delete(note.id)


## Validation

Empty or whitespace-only titles and bodies raise `ValueError`:

python
store.add(title="", body="Something")  # Raises ValueError


## Testing

Run tests with pytest:

bash
pytest


## Constraints

- Uses only the Python standard library
- No external dependencies
- In-memory storage only (no persistence)
