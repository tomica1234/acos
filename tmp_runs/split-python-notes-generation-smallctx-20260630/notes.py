from dataclasses import dataclass, field
from typing import List


@dataclass
class Note:
    """A simple note with title, body, and done status."""
    title: str
    body: str
    done: bool = False

    def __str__(self) -> str:
        status = "[x]" if self.done else "[ ]"
        return f"{status} {self.title}: {self.body}"


class NotesStore:
    """In-memory store for Note objects, keyed by integer ID."""

    def __init__(self) -> None:
        self._notes: dict[int, Note] = {}
        self._next_id: int = 1

    def add(self, note: Note) -> int:
        """Add a note and return its assigned ID."""
        note_id = self._next_id
        self._next_id += 1
        self._notes[note_id] = note
        return note_id

    def list_all(self) -> list[Note]:
        """Return all stored notes in insertion order."""
        return list(self._notes.values())

    def mark_done(self, note_id: int) -> None:
        """Mark a note as done by its ID."""
        if note_id not in self._notes:
            raise ValueError(f"Note with id {note_id} not found")
        self._notes[note_id].done = True

    def delete(self, note_id: int) -> None:
        """Remove a note by its ID."""
        if note_id not in self._notes:
            raise ValueError(f"Note with id {note_id} not found")
        del self._notes[note_id]
