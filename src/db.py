import os
import psycopg

SCHEMA = """
create table if not exists papers (
    arxiv_id text primary key,
    title text not null,
    abstract text not null,
    authors text[] not null,
    categories text[] not null,
    published timestamptz not null,
    hf_upvotes int,
    ingested_at timestamptz not null default now()
);
create table if not exists feedback (
    update_id bigint primary key,
    arxiv_id text not null,
    label text not null,
    created_at timestamptz not null default now()
);
alter table papers add column if not exists embedding real[];
alter table papers add column if not exists features real[];
alter table papers add column if not exists selected_at timestamptz;
create table if not exists models (
    id serial primary key,
    weights real[] not null,
    bias real not null,
    auc real,
    n_labels int not null,
    active boolean not null default false,
    created_at timestamptz not null default now()
);
create unique index if not exists one_active_model on models (active) where active;
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"], prepare_threshold=None)
    conn.execute(SCHEMA)
    return conn
