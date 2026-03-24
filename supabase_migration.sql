-- ResumeForge Web — Database Schema
-- Run this in the Supabase SQL Editor to set up your tables.

-- Runs table: telemetry for every pipeline execution
create table if not exists runs (
    id uuid primary key,
    created_at timestamptz default now(),
    jd_snippet text,
    jd_analysis jsonb,
    critic_eval jsonb,
    page_fit_attempts int default 0,
    overall_score float,
    duration_ms int,
    error text
);

-- Feedback table: user-submitted feedback
create table if not exists feedback (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz default now(),
    run_id uuid references runs(id),
    message text not null,
    rating int check (rating >= 1 and rating <= 5)
);

-- Enable RLS but allow anon/service key full access
alter table runs enable row level security;
alter table feedback enable row level security;

create policy "Allow insert runs" on runs for insert with check (true);
create policy "Allow select runs" on runs for select using (true);
create policy "Allow insert feedback" on feedback for insert with check (true);
create policy "Allow select feedback" on feedback for select using (true);

-- Indexes for common queries
create index idx_runs_created_at on runs(created_at desc);
create index idx_feedback_created_at on feedback(created_at desc);
create index idx_feedback_run_id on feedback(run_id);
create index idx_runs_overall_score on runs(overall_score);
