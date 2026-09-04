"""add attributions table for XAI

Revision ID: 0002_attributions
Revises: 0001_initial
Create Date: 2026-09-04
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0002_attributions"
down_revision: Union[str, Sequence[str], None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "attributions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("layer", sa.String(length=64), nullable=False),
        sa.Column("method", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=256), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column(
            "tokens",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_attributions_request_id", "attributions", ["request_id"])


def downgrade() -> None:
    op.drop_index("ix_attributions_request_id", table_name="attributions")
    op.drop_table("attributions")
