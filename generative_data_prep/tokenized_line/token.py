from typing import Optional

from generative_data_prep.utils import TokenTypeIds


class Token:
    def __init__(self, token_id: int, token_type_id: TokenTypeIds, category_id: Optional[int] = -1):
        self.token_id = token_id
        self.token_type_id = token_type_id
        self.category_id = category_id

    def make_article_boundary(self):
        # assert that this is an eos token
        self.token_type_id = TokenTypeIds.SEP

    def __eq__(self, obj: object) -> bool:
        """Return whether or not another TokenizedLine is equal to this one."""
        if not isinstance(obj, Token):
            return False
        return (
            self.token_id == obj.token_id
            and self.token_type_id == obj.token_type_id
            and self.category_id == obj.category_id
        )

    def __str__(self) -> str:
        """Return the tokenized line as a string."""
        return f"({self.token_id}, {self.token_type_id}, {self.category_id})"

    def __repr__(self) -> str:
        """Return the tokenized line representation.

        Currently the string representation of the tokenized line uniquely identifies a tokenized line, so we just
        call the string function here.
        """
        return str(self)
