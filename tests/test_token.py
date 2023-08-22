import pytest

from generative_data_prep.tokenized_line import Token
from generative_data_prep.utils import TokenTypeIds


@pytest.fixture
def token(
    token_id: int,
    token_type_id: TokenTypeIds,
) -> Token:
    """Creates a token."""
    return Token(token_id, token_type_id)


@pytest.fixture
def token_2(
    token_id_2: int,
    token_type_id_2: TokenTypeIds,
) -> Token:
    """Creates a token."""
    return Token(token_id_2, token_type_id_2)


@pytest.mark.fast
@pytest.mark.parametrize(
    "token_id,token_type_id",
    [(1, TokenTypeIds.COMPLETION), (1, TokenTypeIds.PADDING), (1, TokenTypeIds.PROMPT), (1, TokenTypeIds.SEP)],
)
def test_token_make_article_boundary(token: Token):
    """Test that one tokenized line can be added to another tokenized line."""
    token.make_article_boundary()
    assert token.token_type_id == TokenTypeIds.SEP


@pytest.mark.fast
@pytest.mark.parametrize(
    "token_id,token_type_id,token_id_2,token_type_id_2",
    [(1, TokenTypeIds.COMPLETION, 1, TokenTypeIds.COMPLETION), (2, TokenTypeIds.SEP, 2, TokenTypeIds.SEP)],
)
def test_token_equal(token: Token, token_2: Token):
    """Test that one tokenized line can be added to another tokenized line."""
    assert token == token_2


@pytest.mark.fast
@pytest.mark.parametrize(
    "token_id,token_type_id,token_id_2,token_type_id_2",
    [(2, TokenTypeIds.COMPLETION, 1, TokenTypeIds.COMPLETION), (2, TokenTypeIds.PROMPT, 2, TokenTypeIds.SEP)],
)
def test_token_not_equal(token: Token, token_2: Token):
    """Test that one tokenized line can be added to another tokenized line."""
    assert not token == token_2
