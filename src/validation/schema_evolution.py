"""
Schema Evolution Tracking System for RILA Pipeline.

This module provides comprehensive schema evolution tracking and management
capabilities, detecting schema changes over time and providing guidance
for production ML engineers on handling data evolution.

Key Features:
- Automatic schema fingerprinting and comparison
- Schema evolution history tracking
- Breaking change detection and alerts
- Backward compatibility analysis
- Schema migration recommendations

Following UNIFIED_CODING_STANDARDS.md principles for maintainable evolution tracking.
"""

import pandas as pd
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Defensive imports for optional validation modules
try:
    from src.validation.data_schemas import (
        FINAL_DATASET_SCHEMA, FORECAST_RESULTS_SCHEMA,
        FEATURE_SELECTION_RESULTS_SCHEMA, DataFrameValidator
    )
    from src.config.mlflow_config import safe_mlflow_log_param, safe_mlflow_log_metric
    VALIDATION_MODULES_AVAILABLE = True
except ImportError:
    VALIDATION_MODULES_AVAILABLE = False


class SchemaChangeType(Enum):
    """Types of schema changes detected during evolution."""
    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    COLUMN_TYPE_CHANGED = "column_type_changed"
    COLUMN_NULLABLE_CHANGED = "column_nullable_changed"
    BUSINESS_RULE_CHANGED = "business_rule_changed"
    NO_CHANGE = "no_change"


@dataclass
class SchemaFingerprint:
    """
    Comprehensive fingerprint of a DataFrame schema.

    This captures all relevant schema information for comparison and evolution tracking.
    """
    timestamp: str
    shape: Tuple[int, int]
    columns: List[str]
    column_types: Dict[str, str]
    null_counts: Dict[str, int]
    null_percentages: Dict[str, float]
    memory_usage_mb: float
    duplicate_rows: int
    date_range_days: Optional[int]
    schema_hash: str
    business_rules: List[Dict[str, Any]]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, schema_name: str = "unknown") -> 'SchemaFingerprint':
        """
        Create schema fingerprint from DataFrame.

        Parameters:
            df: DataFrame to fingerprint
            schema_name: Name of the schema context

        Returns:
            SchemaFingerprint object
        """
        # Basic statistics
        shape = df.shape
        columns = list(df.columns)
        column_types = {col: str(df[col].dtype) for col in df.columns}
        null_counts = df.isnull().sum().to_dict()
        null_percentages = {col: (count / len(df) * 100) if len(df) > 0 else 0
                           for col, count in null_counts.items()}
        memory_usage_mb = round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        duplicate_rows = int(df.duplicated().sum())

        # Date range analysis
        date_range_days = None
        if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
            try:
                date_range = df['date'].max() - df['date'].min()
                date_range_days = date_range.days
            except (AttributeError, TypeError, ValueError) as e:
                # Expected failures: empty date column, NaT values, invalid datetime operations
                logger.debug(f"Date range calculation failed: {e}")
                date_range_days = None

        # Business rules (basic heuristics)
        business_rules = cls._extract_business_rules(df)

        # Generate schema hash
        schema_signature = {
            'columns': sorted(columns),
            'types': dict(sorted(column_types.items())),
            'shape': shape
        }
        schema_hash = hashlib.md5(
            json.dumps(schema_signature, sort_keys=True).encode()
        ).hexdigest()[:12]  # Short hash for readability

        return cls(
            timestamp=datetime.now().isoformat(),
            shape=shape,
            columns=columns,
            column_types=column_types,
            null_counts=null_counts,
            null_percentages=null_percentages,
            memory_usage_mb=memory_usage_mb,
            duplicate_rows=duplicate_rows,
            date_range_days=date_range_days,
            schema_hash=schema_hash,
            business_rules=business_rules
        )

    @staticmethod
    def _extract_business_rules(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract inferred business rules from DataFrame."""
        rules = []

        for col in df.columns:
            col_rules = {"column": col, "rules": []}

            # Check for non-negativity (common business rule)
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min()
                max_val = df[col].max()

                if min_val >= 0:
                    col_rules["rules"].append({"type": "non_negative", "min_value": float(min_val)})

                # Check for reasonable ranges
                if max_val > 0 and min_val >= 0:
                    col_rules["rules"].append({
                        "type": "range",
                        "min_value": float(min_val),
                        "max_value": float(max_val)
                    })

            # Check for required fields (no nulls)
            if df[col].isnull().sum() == 0:
                col_rules["rules"].append({"type": "required", "nullable": False})

            if col_rules["rules"]:
                rules.append(col_rules)

        return rules


class SchemaEvolutionTracker:
    """
    Comprehensive schema evolution tracking and management system.

    This class handles schema comparison, evolution detection, and provides
    guidance for production ML engineers on handling schema changes.
    """

    def __init__(self, tracking_dir: str = "schema_evolution"):
        """
        Initialize schema evolution tracker.

        Parameters:
            tracking_dir: Directory to store evolution tracking data
        """
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.tracking_dir / "schema_evolution_history.json"
        self.current_schemas_file = self.tracking_dir / "current_schemas.json"

    def track_schema_evolution(self,
                              df: pd.DataFrame,
                              schema_name: str,
                              baseline_schema: Optional[Any] = None) -> Dict[str, Any]:
        """
        Track schema evolution for a DataFrame.

        Parameters:
            df: DataFrame to track
            schema_name: Name identifier for the schema
            baseline_schema: Optional baseline schema for comparison

        Returns:
            Dict containing evolution analysis results
        """
        print(f"ANALYSIS: Tracking schema evolution for '{schema_name}'...")

        # Generate current schema fingerprint
        current_fingerprint = SchemaFingerprint.from_dataframe(df, schema_name)

        # Load evolution history
        evolution_history = self._load_evolution_history()

        # Get previous fingerprint if available
        previous_fingerprint = None
        if schema_name in evolution_history:
            recent_entries = evolution_history[schema_name]
            if recent_entries:
                # Get most recent fingerprint
                latest_entry = max(recent_entries, key=lambda x: x['timestamp'])
                previous_fingerprint = SchemaFingerprint(**latest_entry['fingerprint'])

        # Perform evolution analysis
        evolution_analysis = self._analyze_schema_changes(
            current_fingerprint, previous_fingerprint, baseline_schema
        )

        # Update history
        self._update_evolution_history(schema_name, current_fingerprint, evolution_analysis)

        # Log to MLflow if available
        self._log_evolution_to_mlflow(schema_name, evolution_analysis)

        print(f"SUCCESS: Schema evolution tracking complete for '{schema_name}'")
        print(f"   Schema Hash: {current_fingerprint.schema_hash}")
        print(f"   Change Status: {evolution_analysis['change_summary']['overall_status']}")

        return evolution_analysis

    def _detect_column_changes(
        self,
        current: SchemaFingerprint,
        previous: SchemaFingerprint
    ) -> Tuple[List[Dict[str, Any]], set, set, set]:
        """Detect added and removed columns between fingerprints.

        Returns:
            Tuple of (changes_list, added_cols, removed_cols, common_cols)
        """
        changes = []
        current_cols = set(current.columns)
        previous_cols = set(previous.columns)

        added_cols = current_cols - previous_cols
        for col in added_cols:
            changes.append({
                "type": SchemaChangeType.COLUMN_ADDED.value,
                "column": col,
                "description": f"Column '{col}' added to schema",
                "breaking": False,
                "data_type": current.column_types.get(col)
            })

        removed_cols = previous_cols - current_cols
        for col in removed_cols:
            changes.append({
                "type": SchemaChangeType.COLUMN_REMOVED.value,
                "column": col,
                "description": f"Column '{col}' removed from schema",
                "breaking": True,
                "previous_data_type": previous.column_types.get(col)
            })

        common_cols = current_cols & previous_cols
        return changes, added_cols, removed_cols, common_cols

    def _detect_type_changes(
        self,
        current: SchemaFingerprint,
        previous: SchemaFingerprint,
        common_cols: set
    ) -> List[Dict[str, Any]]:
        """Detect type changes in common columns."""
        changes = []
        for col in common_cols:
            current_type = current.column_types[col]
            previous_type = previous.column_types[col]

            if current_type != previous_type:
                changes.append({
                    "type": SchemaChangeType.COLUMN_TYPE_CHANGED.value,
                    "column": col,
                    "description": f"Column '{col}' type changed from {previous_type} to {current_type}",
                    "breaking": not self._is_compatible_type_change(previous_type, current_type),
                    "previous_type": previous_type,
                    "current_type": current_type
                })
        return changes

    def _detect_business_rule_changes(
        self,
        current: SchemaFingerprint,
        previous: SchemaFingerprint,
        common_cols: set
    ) -> List[Dict[str, Any]]:
        """Detect business rule changes in common columns."""
        changes = []
        current_rules = {rule["column"]: rule["rules"] for rule in current.business_rules}
        previous_rules = {rule["column"]: rule["rules"] for rule in previous.business_rules}

        for col in common_cols:
            current_col_rules = current_rules.get(col, [])
            previous_col_rules = previous_rules.get(col, [])

            if current_col_rules != previous_col_rules:
                changes.append({
                    "type": SchemaChangeType.BUSINESS_RULE_CHANGED.value,
                    "column": col,
                    "description": f"Business rules changed for column '{col}'",
                    "breaking": self._is_breaking_rule_change(previous_col_rules, current_col_rules),
                    "previous_rules": previous_col_rules,
                    "current_rules": current_col_rules
                })
        return changes

    def _build_change_summary(
        self,
        changes: List[Dict[str, Any]],
        current: SchemaFingerprint,
        previous: SchemaFingerprint,
        added_cols: set,
        removed_cols: set
    ) -> Dict[str, Any]:
        """Build change summary dictionary from detected changes."""
        breaking_changes = sum(1 for change in changes if change.get("breaking", False))
        type_changes = len([c for c in changes if c["type"] == SchemaChangeType.COLUMN_TYPE_CHANGED.value])

        return {
            "overall_status": self._determine_overall_status(changes, breaking_changes),
            "change_count": len(changes),
            "breaking_changes": breaking_changes,
            "schema_hash_changed": current.schema_hash != previous.schema_hash,
            "columns_added": len(added_cols),
            "columns_removed": len(removed_cols),
            "type_changes": type_changes
        }

    def _analyze_schema_changes(
        self,
        current: SchemaFingerprint,
        previous: Optional[SchemaFingerprint],
        baseline_schema: Optional[Any]
    ) -> Dict[str, Any]:
        """Analyze schema changes between fingerprints.

        Orchestrates change detection across columns, types, and business rules.
        """
        analysis = {
            "analysis_timestamp": datetime.now().isoformat(),
            "current_fingerprint": asdict(current),
            "previous_fingerprint": asdict(previous) if previous else None,
            "changes_detected": [],
            "change_summary": {},
            "compatibility_analysis": {},
            "recommendations": []
        }

        if previous is None:
            analysis["change_summary"] = {
                "overall_status": "BASELINE_ESTABLISHED",
                "change_count": 0,
                "breaking_changes": 0,
                "schema_hash_changed": False
            }
            analysis["recommendations"].append("Baseline schema established - future changes will be tracked")
            return analysis

        # Detect all changes using helper methods
        col_changes, added_cols, removed_cols, common_cols = self._detect_column_changes(current, previous)
        type_changes = self._detect_type_changes(current, previous, common_cols)
        rule_changes = self._detect_business_rule_changes(current, previous, common_cols)

        changes = col_changes + type_changes + rule_changes
        analysis["changes_detected"] = changes
        analysis["change_summary"] = self._build_change_summary(
            changes, current, previous, added_cols, removed_cols
        )
        analysis["compatibility_analysis"] = self._analyze_compatibility(current, previous, baseline_schema)
        analysis["recommendations"] = self._generate_recommendations(changes, analysis["compatibility_analysis"])

        return analysis

    def _is_compatible_type_change(self, from_type: str, to_type: str) -> bool:
        """Determine if a type change is backward compatible."""
        compatible_changes = {
            ("int32", "int64"): True,
            ("float32", "float64"): True,
            ("int32", "float64"): True,
            ("int64", "float64"): True,
            ("object", "string"): True,
        }

        return compatible_changes.get((from_type, to_type), False)

    def _is_breaking_rule_change(self, previous_rules: List, current_rules: List) -> bool:
        """Determine if business rule changes are breaking."""
        # Heuristic: Adding rules is generally safe, removing rules might be breaking
        prev_rule_types = {rule.get("type") for rule in previous_rules}
        curr_rule_types = {rule.get("type") for rule in current_rules}

        # If required rule is removed, it's breaking
        if "required" in prev_rule_types and "required" not in curr_rule_types:
            return True

        # If range constraints become more restrictive, it might be breaking
        # This is a simplified heuristic - could be enhanced
        return False

    def _determine_overall_status(self, changes: List, breaking_changes: int) -> str:
        """Determine overall status of schema evolution."""
        if not changes:
            return "NO_CHANGES"
        elif breaking_changes > 0:
            return "BREAKING_CHANGES_DETECTED"
        elif len(changes) > 5:
            return "MAJOR_CHANGES"
        else:
            return "MINOR_CHANGES"

    def _analyze_compatibility(self,
                              current: SchemaFingerprint,
                              previous: SchemaFingerprint,
                              baseline_schema: Optional[Any]) -> Dict[str, Any]:
        """Analyze compatibility with existing systems."""
        compatibility = {
            "backward_compatible": True,
            "forward_compatible": True,
            "baseline_compliant": None,
            "compatibility_score": 0.0,
            "issues": []
        }

        # Basic backward compatibility check
        current_cols = set(current.columns)
        previous_cols = set(previous.columns)

        removed_cols = previous_cols - current_cols
        if removed_cols:
            compatibility["backward_compatible"] = False
            compatibility["issues"].append(f"Removed columns: {list(removed_cols)}")

        # Calculate compatibility score
        common_cols = len(current_cols & previous_cols)
        total_unique_cols = len(current_cols | previous_cols)
        compatibility["compatibility_score"] = common_cols / total_unique_cols if total_unique_cols > 0 else 1.0

        # Baseline schema compliance (if available)
        if baseline_schema and VALIDATION_MODULES_AVAILABLE:
            try:
                # Create temporary DataFrame for validation
                temp_df = pd.DataFrame({col: [None] for col in current.columns})
                baseline_schema.validate(temp_df, lazy=True)
                compatibility["baseline_compliant"] = True
            except Exception as e:
                compatibility["baseline_compliant"] = False
                compatibility["issues"].append(f"Baseline schema compliance failed: {str(e)}")

        return compatibility

    def _generate_recommendations(self,
                                changes: List[Dict],
                                compatibility: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations for handling schema evolution."""
        recommendations = []

        # Breaking changes
        breaking_changes = [c for c in changes if c.get("breaking", False)]
        if breaking_changes:
            recommendations.append(
                "WARNING: BREAKING CHANGES DETECTED - Update dependent systems and tests"
            )
            recommendations.append(
                "Consider implementing schema migration strategy before deploying"
            )

        # Column additions
        added_cols = [c for c in changes if c["type"] == SchemaChangeType.COLUMN_ADDED.value]
        if added_cols:
            recommendations.append(
                f"SUCCESS: {len(added_cols)} new columns added - Update feature engineering if needed"
            )

        # Column removals
        removed_cols = [c for c in changes if c["type"] == SchemaChangeType.COLUMN_REMOVED.value]
        if removed_cols:
            recommendations.append(
                f"ERROR: {len(removed_cols)} columns removed - Check model dependencies"
            )

        # Compatibility score
        compat_score = compatibility.get("compatibility_score", 1.0)
        if compat_score < 0.8:
            recommendations.append(
                f"WARNING: Low compatibility score ({compat_score:.1%}) - Consider schema versioning"
            )

        # General recommendations
        if changes:
            recommendations.extend([
                "Update data validation schemas to match current data structure",
                "Run regression tests to ensure model performance",
                "Update documentation and ML engineer handoff materials"
            ])

        if not recommendations:
            recommendations.append("SUCCESS: No schema changes detected - all systems compatible")

        return recommendations

    def _load_evolution_history(self) -> Dict[str, List]:
        """Load schema evolution history from storage."""
        if not self.history_file.exists():
            return {}

        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"WARNING: Could not load evolution history: {e}")
            return {}

    def _update_evolution_history(self,
                                 schema_name: str,
                                 fingerprint: SchemaFingerprint,
                                 analysis: Dict[str, Any]) -> None:
        """Update evolution history with new fingerprint and analysis."""
        history = self._load_evolution_history()

        if schema_name not in history:
            history[schema_name] = []

        # Add new entry
        history[schema_name].append({
            "timestamp": fingerprint.timestamp,
            "fingerprint": asdict(fingerprint),
            "evolution_analysis": analysis
        })

        # Keep only last 50 entries per schema
        history[schema_name] = history[schema_name][-50:]

        # Save updated history
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
        except Exception as e:
            print(f"WARNING: Could not save evolution history: {e}")

    def _log_evolution_to_mlflow(self, schema_name: str, analysis: Dict[str, Any]) -> None:
        """Log schema evolution metrics to MLflow if available."""
        if not VALIDATION_MODULES_AVAILABLE:
            return

        try:
            # Log key evolution metrics
            change_summary = analysis.get("change_summary", {})

            safe_mlflow_log_param(f"schema_{schema_name}_evolution_status",
                                change_summary.get("overall_status", "UNKNOWN"))
            safe_mlflow_log_metric(f"schema_{schema_name}_change_count",
                                 float(change_summary.get("change_count", 0)))
            safe_mlflow_log_metric(f"schema_{schema_name}_breaking_changes",
                                 float(change_summary.get("breaking_changes", 0)))

            compatibility = analysis.get("compatibility_analysis", {})
            safe_mlflow_log_metric(f"schema_{schema_name}_compatibility_score",
                                 compatibility.get("compatibility_score", 1.0))

        except Exception as e:
            print(f"WARNING: Could not log to MLflow: {e}")

    def get_evolution_summary(self, schema_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive evolution summary for schemas.

        Parameters:
            schema_name: Optional specific schema name, or None for all schemas

        Returns:
            Dict containing evolution summary
        """
        history = self._load_evolution_history()

        if schema_name:
            schema_history = history.get(schema_name, [])
            return self._generate_schema_summary(schema_name, schema_history)
        else:
            summary = {
                "generation_timestamp": datetime.now().isoformat(),
                "schemas_tracked": list(history.keys()),
                "total_schemas": len(history),
                "schema_summaries": {}
            }

            for name, schema_history in history.items():
                summary["schema_summaries"][name] = self._generate_schema_summary(name, schema_history)

            return summary

    def _generate_schema_summary(self, schema_name: str, schema_history: List) -> Dict[str, Any]:
        """Generate summary for a specific schema's evolution history."""
        if not schema_history:
            return {"status": "NO_HISTORY", "entries": 0}

        latest_entry = max(schema_history, key=lambda x: x['timestamp'])

        total_changes = sum(
            entry["evolution_analysis"]["change_summary"]["change_count"]
            for entry in schema_history
            if "evolution_analysis" in entry
        )

        return {
            "status": "TRACKED",
            "entries": len(schema_history),
            "latest_timestamp": latest_entry["timestamp"],
            "latest_schema_hash": latest_entry["fingerprint"]["schema_hash"],
            "total_changes_tracked": total_changes,
            "latest_shape": latest_entry["fingerprint"]["shape"],
            "latest_columns": len(latest_entry["fingerprint"]["columns"])
        }


# Convenience functions for notebook/pipeline usage
def track_dataset_evolution(df: pd.DataFrame,
                          dataset_name: str,
                          tracker: Optional[SchemaEvolutionTracker] = None) -> Dict[str, Any]:
    """
    Convenience function to track dataset evolution.

    Parameters:
        df: DataFrame to track
        dataset_name: Name of the dataset
        tracker: Optional existing tracker instance

    Returns:
        Evolution analysis results
    """
    if tracker is None:
        tracker = SchemaEvolutionTracker()

    return tracker.track_schema_evolution(df, dataset_name)


def get_evolution_report(schema_name: Optional[str] = None,
                        tracker: Optional[SchemaEvolutionTracker] = None) -> Dict[str, Any]:
    """
    Convenience function to get evolution report.

    Parameters:
        schema_name: Optional specific schema name
        tracker: Optional existing tracker instance

    Returns:
        Evolution report
    """
    if tracker is None:
        tracker = SchemaEvolutionTracker()

    return tracker.get_evolution_summary(schema_name)


if __name__ == "__main__":
    print("START: Schema Evolution Tracking System Test")
    print("=" * 50)

    # Test with sample data evolution
    print("\nREPORT: Testing schema evolution with sample data...")

    # Create baseline dataset
    baseline_data = pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=100),
        'sales': np.random.uniform(50000, 200000, 100),
        'price': np.random.uniform(1, 5, 100)
    })

    # Track baseline
    tracker = SchemaEvolutionTracker()
    baseline_analysis = tracker.track_schema_evolution(baseline_data, "sample_dataset")
    print(f"SUCCESS: Baseline tracking: {baseline_analysis['change_summary']['overall_status']}")

    # Create evolved dataset (add column, change type)
    evolved_data = baseline_data.copy()
    evolved_data['new_feature'] = np.random.randn(100)
    evolved_data['sales'] = evolved_data['sales'].astype('int64')  # Type change

    # Track evolution
    evolution_analysis = tracker.track_schema_evolution(evolved_data, "sample_dataset")
    print(f"SUCCESS: Evolution tracking: {evolution_analysis['change_summary']['overall_status']}")
    print(f"   Changes detected: {evolution_analysis['change_summary']['change_count']}")

    # Generate evolution report
    report = tracker.get_evolution_summary()
    print(f"\n[REPORT] Evolution Report Generated:")
    print(f"   Schemas tracked: {report['total_schemas']}")

    print("\nSUCCESS: Schema evolution tracking system test completed!")