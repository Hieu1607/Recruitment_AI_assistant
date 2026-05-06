"""add oauth_identities table

Revision ID: a06907d540de
Revises: 20260429_0002
Create Date: 2026-05-04 15:31:16.230332

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a06907d540de'
down_revision = '20260429_0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('oauth_identities',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('provider', sa.String(length=32), nullable=False),
    sa.Column('provider_subject', sa.String(length=255), nullable=False),
    sa.Column('email', sa.String(length=320), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user_accounts.id'], name=op.f('fk_oauth_identities_user_accounts'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_oauth_identities')),
    sa.UniqueConstraint('provider', 'provider_subject', name='uq_oauth_identity_provider_subject')
    )
    op.create_index(op.f('ix_oauth_identities_user_id'), 'oauth_identities', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_oauth_identities_user_id'), table_name='oauth_identities')
    op.drop_table('oauth_identities')
