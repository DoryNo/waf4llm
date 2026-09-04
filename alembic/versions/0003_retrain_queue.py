"""add retrain_queue table for continuous retraining

Revision ID: 0003_retrain_queue
Revises: 0002_attributions
Create Date: 2026-09-05
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0003_retrain_queue"
down_revision: Union[str, Sequence[str], None] = "0002_attributions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "retrain_queue",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("label", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("request_id", sa.String(length=36), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("decision", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "meta",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_retrain_queue_status", "retrain_queue", ["status"])
    op.create_index("ix_retrain_queue_created_at", "retrain_queue", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_retrain_queue_created_at", table_name="retrain_queue")
    op.drop_index("ix_retrain_queue_status", table_name="retrain_queue")
    op.drop_table("retrain_queue")
