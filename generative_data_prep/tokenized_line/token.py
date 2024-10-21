"""Copyright 2023 SambaNova Systems, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Optional

from generative_data_prep.utils import TokenTypeIds


class Token:
    """Class to represent a token, and all the metadata associated with it."""

    def __init__(self, token_id: int, token_type_id: TokenTypeIds, category_id: Optional[int] = -1):
        """Initialize a token object.

        Args:
            token_id: The token id for this token.
            token_type_id: The token type id of this token.
            category_id: The category which this token belongs to, user defined metadata for plotting loss curves.
        """
        self.token_id = token_id
        self.token_type_id = token_type_id
        self.category_id = category_id

    def make_article_boundary(self):
        """Turn this token into an article attention boundary."""
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
        return f"({self.token_id}, {self.token_type_id})"

    def __repr__(self) -> str:
        """Return the tokenized line representation.

        Currently the string representation of the tokenized line uniquely identifies a tokenized line, so we just
        call the string function here.
        """
        return str(self)
