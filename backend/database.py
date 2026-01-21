"""
SmartRoot AI - Supabase Database Integration
Handles persistent storage for dashboard stats and analysis history
"""

import streamlit as st
from datetime import datetime
from supabase import create_client

# -------------------------------------------------
# SUPABASE CLIENT INITIALIZATION
# -------------------------------------------------

_supabase_client = None

def get_supabase_client():
    """Initialize and cache Supabase client"""
    global _supabase_client
    
    if _supabase_client is not None:
        return _supabase_client
    
    try:
        url = st.secrets.get("supabase", {}).get("url")
        key = st.secrets.get("supabase", {}).get("key")
        
        if not url or not key:
            return None
        
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as e:
        st.warning(f"Supabase connection failed: {e}. Using session storage.")
        return None


def init_database():
    """
    Initialize database tables if they don't exist.
    Run this SQL in Supabase SQL Editor:
    
    -- Dashboard Stats Table
    CREATE TABLE IF NOT EXISTS dashboard_stats (
        id SERIAL PRIMARY KEY,
        total_plants INTEGER DEFAULT 0,
        total_roots INTEGER DEFAULT 0,
        total_simulations INTEGER DEFAULT 0,
        avg_health_score FLOAT DEFAULT 0,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    -- Analysis History Table
    CREATE TABLE IF NOT EXISTS analysis_history (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        icon VARCHAR(10) NOT NULL,
        health_score INTEGER NOT NULL,
        analysis_type VARCHAR(50) NOT NULL,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    -- Health Metrics Table (for trend charts)
    CREATE TABLE IF NOT EXISTS health_metrics (
        id SERIAL PRIMARY KEY,
        health_score INTEGER,
        moisture INTEGER,
        nutrient INTEGER,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    -- Insert initial dashboard stats row
    INSERT INTO dashboard_stats (id, total_plants, total_roots, total_simulations, avg_health_score)
    VALUES (1, 0, 0, 0, 0)
    ON CONFLICT (id) DO NOTHING;
    """
    pass


# -------------------------------------------------
# DASHBOARD STATS OPERATIONS
# -------------------------------------------------

def get_dashboard_stats():
    """Fetch dashboard stats from database"""
    supabase = get_supabase_client()
    if not supabase:
        return None
    
    try:
        response = supabase.table('dashboard_stats').select('*').eq('id', 1).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        print(f"Error fetching dashboard stats: {e}")
        return None


def update_dashboard_stats_db(stats_update: dict):
    """Update dashboard stats in database"""
    supabase = get_supabase_client()
    if not supabase:
        return False
    
    try:
        stats_update['updated_at'] = datetime.now().isoformat()
        response = supabase.table('dashboard_stats').update(stats_update).eq('id', 1).execute()
        return bool(response.data)
    except Exception as e:
        print(f"Error updating dashboard stats: {e}")
        return False


def increment_stat(stat_name: str, increment: int = 1):
    """Increment a specific stat counter"""
    supabase = get_supabase_client()
    if not supabase:
        return False
    
    try:
        # Fetch current value
        current = get_dashboard_stats()
        if current:
            new_value = current.get(stat_name, 0) + increment
            return update_dashboard_stats_db({stat_name: new_value})
        return False
    except Exception as e:
        print(f"Error incrementing stat: {e}")
        return False


# -------------------------------------------------
# ANALYSIS HISTORY OPERATIONS
# -------------------------------------------------

def add_analysis_to_db(name: str, icon: str, health_score: int, analysis_type: str):
    """Add a new analysis record to history"""
    supabase = get_supabase_client()
    if not supabase:
        return False
    
    try:
        record = {
            'name': name,
            'icon': icon,
            'health_score': health_score,
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat()
        }
        response = supabase.table('analysis_history').insert(record).execute()
        return bool(response.data)
    except Exception as e:
        print(f"Error adding analysis: {e}")
        return False


def get_analysis_history(limit: int = 10):
    """Fetch recent analysis history"""
    supabase = get_supabase_client()
    if not supabase:
        return []
    
    try:
        response = supabase.table('analysis_history').select('*').order('timestamp', desc=True).limit(limit).execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []


# -------------------------------------------------
# HEALTH METRICS (FOR TREND CHARTS)
# -------------------------------------------------

def add_health_metric(health_score: int, moisture: int, nutrient: int):
    """Add health metric for trend tracking"""
    supabase = get_supabase_client()
    if not supabase:
        return False
    
    try:
        record = {
            'health_score': health_score,
            'moisture': moisture,
            'nutrient': nutrient,
            'timestamp': datetime.now().isoformat()
        }
        response = supabase.table('health_metrics').insert(record).execute()
        return bool(response.data)
    except Exception as e:
        print(f"Error adding metric: {e}")
        return False


def get_health_metrics(limit: int = 20):
    """Fetch health metrics for trend chart"""
    supabase = get_supabase_client()
    if not supabase:
        return []
    
    try:
        response = supabase.table('health_metrics').select('*').order('timestamp', desc=True).limit(limit).execute()
        # Reverse to get chronological order
        return list(reversed(response.data)) if response.data else []
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return []


# -------------------------------------------------
# SYNC SESSION STATE WITH DATABASE
# -------------------------------------------------

def sync_stats_to_session():
    """Sync database stats to session state"""
    db_stats = get_dashboard_stats()
    if db_stats:
        st.session_state.dashboard_stats['total_plants'] = db_stats.get('total_plants', 0)
        st.session_state.dashboard_stats['total_roots'] = db_stats.get('total_roots', 0)
        st.session_state.dashboard_stats['total_simulations'] = db_stats.get('total_simulations', 0)
        st.session_state.dashboard_stats['avg_health_score'] = db_stats.get('avg_health_score', 0)
    
    # Sync health metrics for charts
    metrics = get_health_metrics()
    if metrics:
        st.session_state.dashboard_stats['timestamps'] = [m['timestamp'] for m in metrics]
        st.session_state.dashboard_stats['health_scores'] = [m['health_score'] for m in metrics]
        st.session_state.dashboard_stats['moisture_history'] = [m['moisture'] for m in metrics]
        st.session_state.dashboard_stats['nutrient_history'] = [m['nutrient'] for m in metrics]
    
    # Sync analysis history
    history = get_analysis_history()
    if history:
        st.session_state.analysis_history = [
            {
                'name': h['name'],
                'icon': h['icon'],
                'health_score': h['health_score'],
                'analysis_type': h['analysis_type'],
                'timestamp': h['timestamp']
            }
            for h in history
        ]


def save_analysis_to_db(name: str, icon: str, health_score: int, analysis_type: str, moisture: int = 50, nutrient: int = 50):
    """Save analysis and update all related stats"""
    # Add to analysis history
    add_analysis_to_db(name, icon, health_score, analysis_type)
    
    # Add health metric
    add_health_metric(health_score, moisture, nutrient)
    
    # Update counters
    db_stats = get_dashboard_stats()
    if db_stats:
        updates = {}
        
        if analysis_type == 'plant':
            updates['total_plants'] = db_stats.get('total_plants', 0) + 1
        elif analysis_type == 'root':
            updates['total_roots'] = db_stats.get('total_roots', 0) + 1
        elif analysis_type == 'simulation':
            updates['total_simulations'] = db_stats.get('total_simulations', 0) + 1
        
        # Recalculate average health score
        metrics = get_health_metrics(100)  # Get last 100 for average
        if metrics:
            avg_score = sum(m['health_score'] for m in metrics) / len(metrics)
            updates['avg_health_score'] = round(avg_score, 1)
        
        if updates:
            update_dashboard_stats_db(updates)
