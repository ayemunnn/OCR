"""add google auth fields

Revision ID: 20260523_0002
Revises: 20260515_0001
Create Date: 2026-05-23
"""

from alembic import op
import sqlalchemy as sa


revision = "20260523_0002"
down_revision = "20260515_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "auth_provider",
            sa.String(),
            nullable=False,
            server_default="email",
        ),
    )
    op.add_column("users", sa.Column("google_sub", sa.String(), nullable=True))
    op.add_column(
        "users",
        sa.Column(
            "email_verified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.create_index(op.f("ix_users_google_sub"), "users", ["google_sub"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_users_google_sub"), table_name="users")
    op.drop_column("users", "email_verified")
    op.drop_column("users", "google_sub")
    op.drop_column("users", "auth_provider")
