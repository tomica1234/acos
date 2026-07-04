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

    def add(self, title: str, body: str) -> Note:
        """Add a new note and return it with a generated ID.

        Raises:
            ValueError: If title or body is empty or whitespace-only.
        """
        if not title or not title.strip():
            raise ValueError("Title cannot be empty")
        if not body or not body.strip():
            raise ValueError("Body cannot be empty")
        note = Note(title=title, body=body)
        self._notes[note.id] = note
        return note

    def list_all(self) -> List[Note]:
        """Return a list of all notes."""
        return list(self._notes.values())

    def mark_done(self, note_id: str) -> Note:
        """Mark a note as done by its ID.

        Raises:
            ValueError: If no note with the given ID exists.
        """
        if note_id not in self._notes:
            raise ValueError(f"No note found with id: {note_id}")
        self._notes[note_id].done = True
        return self._notes[note_id]

    def delete(self, note_id: str) -> Note:
        """Delete a note by its ID.

        Raises:
            ValueError: If no note with the given ID exists.
        """
        if note_id not in self._notes:
            raise ValueError(f"No note found with id: {note_id}")
        return self._notes.pop(note_id)
