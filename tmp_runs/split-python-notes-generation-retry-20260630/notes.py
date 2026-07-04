from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid


@dataclass
class Note:
    """A simple note with title, body, and done status."""
    title: str
    body: str
    done: bool = False
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class NotesStore:
    """In-memory store for Note objects."""

    def __init__(self) -> None:
        self._notes: Dict[str, Note] = {}

    def add(self, title: str, body: str) -> str:
        """Add a new note. Returns the note ID."""
        note = Note(title=title, body=body)
        self._notes[note.id] = note
        return note.id

    def list_all(self) -> List[Note]:
        """Return all notes as a list."""
        return list(self._notes.values())

    def mark_done(self, note_id: str) -> None:
        """Mark a note as done. Raises ValueError if note_id not found."""
        if note_id not in self._notes:
            raise ValueError(f"Note with id '{note_id}' not found.")
        self._notes[note_id].done = True

    def delete(self, note_id: str) -> None:
        """Delete a note. Raises ValueError if note_id not found."""
        if note_id not in self._notes:
            raise ValueError(f"Note with id '{note_id}' not found.")
        del self._notes[note_id]
