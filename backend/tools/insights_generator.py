"""
Insights generator for UAV flight data.
"""

from models import V2ConversationSession


class InsightsGenerator:
    """Generates comprehensive insights about UAV flights."""

    def generate_insights(self, session: V2ConversationSession, focus: str) -> str:
        """Generate comprehensive insights about the flight."""
        try:
            insights = []
            
            # Flight duration
            if any('timestamp' in df.columns for df in session.dataframes.values()):
                timestamps = []
                for df in session.dataframes.values():
                    if 'timestamp' in df.columns:
                        timestamps.extend(df['timestamp'].dropna().tolist())
                
                if timestamps:
                    duration = (max(timestamps) - min(timestamps)).total_seconds() / 60
                    insights.append(f"Flight duration: {duration:.1f} minutes")
            
            # Altitude insights
            if 'GPS' in session.dataframes:
                gps_df = session.dataframes['GPS']
                if 'Alt' in gps_df.columns:
                    max_alt = gps_df['Alt'].max()
                    avg_alt = gps_df['Alt'].mean()
                    insights.append(f"Maximum altitude: {max_alt:.1f}m, Average altitude: {avg_alt:.1f}m")
            
            # Speed insights
            if 'GPS' in session.dataframes:
                gps_df = session.dataframes['GPS']
                if 'Spd' in gps_df.columns:
                    max_speed = gps_df['Spd'].max()
                    avg_speed = gps_df['Spd'].mean()
                    insights.append(f"Maximum speed: {max_speed:.1f} m/s, Average speed: {avg_speed:.1f} m/s")
            
            # Mode changes
            if 'MODE' in session.dataframes:
                mode_df = session.dataframes['MODE']
                mode_changes = len(mode_df)
                insights.append(f"Flight mode changes: {mode_changes}")
            
            if not insights:
                insights.append("Basic flight data processed successfully")
            
            return "Flight Analysis Summary:\n" + "\n".join(f"• {insight}" for insight in insights)
            
        except Exception as e:
            return f"Error generating insights: {str(e)}" 