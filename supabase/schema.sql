-- ROCm-KernelTuner Supabase Schema
-- Fine-Tuning on AMD GPUs track — AMD Developer Hackathon

-- User kernel optimization queries
create table if not exists kernel_queries (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade,
    original_code text not null,
    target text default 'hip',
    gpu text default 'mi300x',
    optimized_code text,
    created_at timestamptz default now()
);

-- XMRig config generations
create table if not exists xmrig_configs (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade,
    gpu text not null,
    threads int,
    pool text,
    wallet text,
    config_json jsonb not null,
    predicted_hashrate int,
    predicted_wattage int,
    created_at timestamptz default now()
);

-- Hashrate benchmark predictions vs reality
create table if not exists benchmark_predictions (
    id uuid primary key default gen_random_uuid(),
    kernel text,
    gpu text not null,
    memory_gb int,
    threads int,
    predicted_hashrate int,
    predicted_wattage int,
    actual_hashrate int,
    actual_wattage int,
    accuracy_percent numeric,
    created_at timestamptz default now()
);

-- User feedback on generations
create table if not exists generation_feedback (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade,
    generation_type text not null,
    rating int check (rating between 1 and 5),
    feedback text,
    created_at timestamptz default now()
);

-- Enable RLS
alter table kernel_queries enable row level security;
alter table xmrig_configs enable row level security;
alter table benchmark_predictions enable row level security;
alter table generation_feedback enable row level security;

-- Policies
 create policy "Users own queries"
    on kernel_queries for all
    using (user_id = auth.uid());

 create policy "Users own configs"
    on xmrig_configs for all
    using (user_id = auth.uid());

 create policy "Users own feedback"
    on generation_feedback for all
    using (user_id = auth.uid());

 create policy "Benchmarks public read"
    on benchmark_predictions for select to anon, authenticated using (true);

-- Indexes
 create index idx_kernel_user on kernel_queries(user_id);
 create index idx_xmrig_user on xmrig_configs(user_id);
 create index idx_benchmark_gpu on benchmark_predictions(gpu);
 create index idx_benchmark_created on benchmark_predictions(created_at desc);
