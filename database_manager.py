import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

class ExperimentDatabase:
    """JSON-based database for storing experimental records"""

    def __init__(self, db_dir: str = 'data/database'):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.sessions_file = self.db_dir / 'sessions.json'
        self.experiments_file = self.db_dir / 'experiments.json'
        self.materials_file = self.db_dir / 'materials.json'
        self.configs_file = self.db_dir / 'configs.json'

        self._initialize_databases()

    def _initialize_databases(self):
        if not self.sessions_file.exists():
            self._save_json(self.sessions_file, {})
        if not self.experiments_file.exists():
            self._save_json(self.experiments_file, [])
        if not self.materials_file.exists():
            self._save_json(self.materials_file, [])
        if not self.configs_file.exists():
            self._save_json(self.configs_file, {})

    def _load_json(self, filepath: Path) -> Any:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            # Corrupted file — attempt recovery
            print(f"Warning: Corrupted JSON in {filepath}, attempting recovery...")
            backup_path = filepath.with_suffix('.json.bak')
            try:
                filepath.rename(backup_path)
            except OSError:
                pass
            with open(backup_path, 'r', encoding='utf-8') as f:
                raw = f.read()

            # For a truncated array file, find the last '}' and close with ']'
            if raw.lstrip().startswith('['):
                last_brace = raw.rfind('}')
                if last_brace > 0:
                    truncated = raw[:last_brace + 1] + ']'
                    try:
                        data = json.loads(truncated)
                        print(f"Recovered {len(data)} records from corrupted {filepath.name}")
                        # Save the repaired file
                        self._save_json(filepath, data)
                        return data
                    except json.JSONDecodeError:
                        pass

            # For a truncated object file, find the last '}' and close with '}'
            if raw.lstrip().startswith('{'):
                last_brace = raw.rfind('}')
                if last_brace > 0:
                    truncated = raw[:last_brace + 1] + '}'
                    try:
                        data = json.loads(truncated)
                        print(f"Recovered data from corrupted {filepath.name}")
                        self._save_json(filepath, data)
                        return data
                    except json.JSONDecodeError:
                        pass

            # Unrecoverable — start fresh
            print(f"Warning: Could not repair {filepath.name}, starting with empty data")
            empty = [] if filepath in (self.experiments_file, self.materials_file) else {}
            self._save_json(filepath, empty)
            return empty

    def _sanitize_for_json(self, obj):
        """Recursively replace NaN/Inf floats with None so json.dump produces valid JSON."""
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return self._sanitize_for_json(obj.tolist())
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_for_json(v) for v in obj]
        if isinstance(obj, datetime):
            return obj.isoformat()
        if obj is pd.NA or (isinstance(obj, float) and pd.isna(obj)):
            return None
        return obj

    def _save_json(self, filepath: Path, data: Any):
        sanitized = self._sanitize_for_json(data)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sanitized, f, indent=2, ensure_ascii=False, allow_nan=False)

    def _json_serializer(self, obj):
        # Handle NaN, Infinity, -Infinity
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None  # Convert NaN/Inf to null
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            # Check for NaN/Inf again for numpy floats
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        else:
            return str(obj)

    # ---------- Session Management ----------
    def create_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        sessions = self._load_json(self.sessions_file)
        if session_id in sessions:
            return False
        session_data['created_at'] = datetime.now().isoformat()
        sessions[session_id] = session_data
        self._save_json(self.sessions_file, sessions)
        return True

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        sessions = self._load_json(self.sessions_file)
        if session_id not in sessions:
            return False
        if 'created_at' not in updates:
            updates['created_at'] = sessions[session_id]['created_at']
        if updates.get('status') == 'completed' and sessions[session_id].get('status') != 'completed':
            updates['completed_at'] = datetime.now().isoformat()
        sessions[session_id].update(updates)
        self._save_json(self.sessions_file, sessions)
        return True

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        sessions = self._load_json(self.sessions_file)
        return sessions.get(session_id)

    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        return self._load_json(self.sessions_file)

    def delete_session(self, session_id: str) -> bool:
        sessions = self._load_json(self.sessions_file)
        if session_id not in sessions:
            return False
        del sessions[session_id]
        self._save_json(self.sessions_file, sessions)
        self._delete_experiments_by_session(session_id)
        return True

    # ---------- Experiment Management ----------
    def add_experiment(self, session_id: str, experiment_data: Dict[str, Any]) -> int:
        experiments = self._load_json(self.experiments_file)
        experiment_id = max((exp.get('experiment_id', 0) for exp in experiments), default=0) + 1
        experiment_data['experiment_id'] = experiment_id
        experiment_data['session_id'] = session_id
        experiment_data['timestamp'] = datetime.now().isoformat()
        experiments.append(experiment_data)
        self._save_json(self.experiments_file, experiments)
        return experiment_id

    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        experiments = self._load_json(self.experiments_file)
        for exp in experiments:
            if exp.get('experiment_id') == experiment_id:
                return exp
        return None

    def get_experiments_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        experiments = self._load_json(self.experiments_file)
        return [exp for exp in experiments if exp.get('session_id') == session_id]

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        return self._load_json(self.experiments_file)

    def _delete_experiments_by_session(self, session_id: str):
        experiments = self._load_json(self.experiments_file)
        experiments = [exp for exp in experiments if exp.get('session_id') != session_id]
        self._save_json(self.experiments_file, experiments)

    # ---------- Material Management ----------
    def add_material(self, material_data: Dict[str, Any]) -> int:
        materials = self._load_json(self.materials_file)
        material_id = max((mat.get('material_id', 0) for mat in materials), default=0) + 1
        material_data['material_id'] = material_id
        material_data['created_at'] = datetime.now().isoformat()
        materials.append(material_data)
        self._save_json(self.materials_file, materials)
        return material_id

    def get_material(self, material_id: int) -> Optional[Dict[str, Any]]:
        materials = self._load_json(self.materials_file)
        for mat in materials:
            if mat.get('material_id') == material_id:
                return mat
        return None

    def get_all_materials(self) -> List[Dict[str, Any]]:
        return self._load_json(self.materials_file)

    # ---------- Configuration Management ----------
    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> bool:
        configs = self._load_json(self.configs_file)
        config_data['saved_at'] = datetime.now().isoformat()
        configs[config_name] = config_data
        self._save_json(self.configs_file, configs)
        return True

    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        configs = self._load_json(self.configs_file)
        return configs.get(config_name)

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        return self._load_json(self.configs_file)

    # ---------- Data Export ----------
    def export_to_csv(self, session_id: Optional[str] = None) -> pd.DataFrame:
        experiments = self.get_experiments_by_session(session_id) if session_id else self.get_all_experiments()
        if not experiments:
            return pd.DataFrame()
        df = pd.DataFrame(experiments)
        if 'candidate' in df.columns:
            candidate_df = pd.json_normalize(df['candidate'])
            df = pd.concat([df.drop(columns=['candidate']), candidate_df], axis=1)
        return df

    def export_to_json(self, session_id: Optional[str] = None) -> str:
        if session_id:
            data = {
                'session': self.get_session(session_id),
                'experiments': self.get_experiments_by_session(session_id)
            }
        else:
            data = {
                'sessions': self.get_all_sessions(),
                'experiments': self.get_all_experiments(),
                'materials': self.get_all_materials()
            }
        return json.dumps(data, indent=2, ensure_ascii=False, default=self._json_serializer)

    # ---------- Backup and Restore ----------
    def backup_database(self, backup_path: str) -> bool:
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_data = {
                'sessions': self._load_json(self.sessions_file),
                'experiments': self._load_json(self.experiments_file),
                'materials': self._load_json(self.materials_file),
                'configs': self._load_json(self.configs_file),
                'backup_timestamp': datetime.now().isoformat()
            }
            backup_file = backup_dir / f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            return True
        except Exception as e:
            print(f"Backup failed: {e}")
            return False

    def restore_from_backup(self, backup_file: str) -> bool:
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            if 'sessions' in backup_data:
                self._save_json(self.sessions_file, backup_data['sessions'])
            if 'experiments' in backup_data:
                self._save_json(self.experiments_file, backup_data['experiments'])
            if 'materials' in backup_data:
                self._save_json(self.materials_file, backup_data['materials'])
            if 'configs' in backup_data:
                self._save_json(self.configs_file, backup_data['configs'])
            return True
        except Exception as e:
            print(f"Restore failed: {e}")
            return False

    # ---------- Reset ----------
    def reset_database(self) -> bool:
        """Delete all data from the database."""
        try:
            self._save_json(self.sessions_file, {})
            self._save_json(self.experiments_file, [])
            self._save_json(self.materials_file, [])
            self._save_json(self.configs_file, {})
            return True
        except Exception as e:
            print(f"Reset failed: {e}")
            return False

    # ---------- Statistics ----------
    def get_statistics(self) -> Dict[str, Any]:
        sessions = self._load_json(self.sessions_file)
        experiments = self._load_json(self.experiments_file)
        materials = self._load_json(self.materials_file)

        active_sessions = sum(1 for s in sessions.values() if s.get('status') == 'running')
        completed_sessions = sum(1 for s in sessions.values() if s.get('status') == 'completed')
        avg_exp_per_session = len(experiments) / max(1, len(sessions))

        best_perf = 0.0
        for exp in experiments:
            perf = exp.get('experimental_performance', 0)
            if perf > best_perf:
                best_perf = perf

        stats = {
            'total_sessions': len(sessions),
            'total_experiments': len(experiments),
            'total_materials': len(materials),
            'active_sessions': active_sessions,
            'completed_sessions': completed_sessions,
            'average_experiments_per_session': avg_exp_per_session,
            'best_performance': best_perf,
            'last_backup': self._get_last_backup_timestamp()
        }
        return stats

    def _get_last_backup_timestamp(self) -> Optional[str]:
        backup_dir = self.db_dir / 'backups'
        if backup_dir.exists():
            backups = list(backup_dir.glob('backup_*.json'))
            if backups:
                latest = max(backups, key=lambda p: p.stat().st_mtime)
                return datetime.fromtimestamp(latest.stat().st_mtime).isoformat()
        return None

    def cleanup_old_sessions(self, days_old: int = 30):
        sessions = self._load_json(self.sessions_file)
        current_time = datetime.now()
        sessions_to_remove = []
        for session_id, session_data in sessions.items():
            created_at = session_data.get('created_at')
            if created_at:
                try:
                    created_time = datetime.fromisoformat(created_at)
                    if (current_time - created_time).days > days_old:
                        sessions_to_remove.append(session_id)
                except ValueError:
                    continue
        for session_id in sessions_to_remove:
            self.delete_session(session_id)