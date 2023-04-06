"""
Copyright 2023 SambaNova Systems, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This module implements user specifiable options (through the command line) for the data preparation pipeline.
"""

from typing import List, Optional, Union

from .constants import OverflowType, PackingStyleType


class PackingConfig:
    """Options to specify how the input articles should be packed into sequences."""

    DELIM = '::'  # delimiter between packing style name and overflow type name in the packing config name.

    def __init__(self,
                 packing_style: Union[PackingStyleType, str],
                 overflow_type: Optional[Union[OverflowType, str]] = None):
        """Create the PackingConfig.

        Args:
            packing_style:  Specifies how to pack the tokens into sequences.  Refer to SequencePacker class.
            overflow_type:  Specifies how to deal with overflow tokens that don't fit into the sequences.

        Raises:
            ValueError: Invalid combination of PackingStyle and OverflowType.  Not all overflow types are compatible
                with all packing styles.
        """
        self._overflow_type = OverflowType(
            overflow_type) if overflow_type is not None else None
        self._packing_style = PackingStyleType(packing_style)

        # error handling
        if self.packing_style is PackingStyleType.FULL and self.overflow_type:
            err_msg_1 = 'No overflow type cannot be specified for "FULL" packing style.'
            err_msg_2 = f'Found overflow type: {self.overflow_type}'
            raise ValueError(f'{err_msg_1} {err_msg_2}')

    @classmethod
    def from_str(cls, packing_config_name: str):
        """Create the PackingConfig from the packing config name.

        Args:
            packing_config_name: The name of the packing config specified by a space separated strings for the
                packing style and overflow type.

        Raises:
            ValueError: Invalid PackingConfig name.

        Returns:
            The newly created PackingConfig.
        """
        # both packing style and overflow type are specified
        if cls.DELIM in packing_config_name:
            split_name = packing_config_name.split(cls.DELIM)

            # error handling
            if len(split_name) != 2:
                err_msg_1 = 'Packing config name must be formatted as "<packing_style> <overflow_type>"'
                err_msg_2 = '<packing_style> and <overflow_type> may not contain any spaces.'
                raise ValueError(f'{err_msg_1} {err_msg_2}')

            packing_style, overflow_type = tuple(split_name)
        # only packing style is specified
        else:
            packing_style, overflow_type = packing_config_name, None

        return cls(packing_style, overflow_type)

    @classmethod
    def get_default(cls) -> 'PackingConfig':
        """Return the default PackingConfig."""
        return cls(PackingStyleType.FULL)

    def __eq__(self, other: object):
        """Check equality by checking if the string representations are the same."""
        if not isinstance(other, PackingConfig):
            return NotImplemented
        return str(self) == str(other)

    def __repr__(self) -> str:
        """See __str__ method documentation."""
        return str(self)

    def __str__(self) -> str:
        """Return a string representation of the PackingConfig."""
        config_tuple = (typ
                        for typ in [self.packing_style, self.overflow_type]
                        if typ is not None)
        return PackingConfig.DELIM.join(config_tuple)

    @staticmethod
    def get_choices() -> List['PackingConfig']:
        """Return the names of the different PackingConfig choices."""
        choices = []
        for packing_style_type in list(PackingStyleType):
            # cannot specify an overflow type for PackingStyle FULL.
            if packing_style_type == PackingStyleType.FULL:
                choices.append(PackingConfig(packing_style_type))
            else:
                for overflow_type in list(OverflowType):
                    choices.append(
                        PackingConfig(packing_style_type, overflow_type))

        return choices

    @property
    def overflow_type(self) -> Optional[OverflowType]:
        """Return the overflow type."""
        return self._overflow_type

    @property
    def packing_style(self) -> PackingStyleType:
        """Return the packing style."""
        return self._packing_style
