from typing import Dict, Any, Optional

from position_inference.config import get_action_role_rules


def match_action_rule(action_name: str, play_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Matches raw action name to configured rule in action_role_rules.yaml.
    """
    rules = get_action_role_rules()
    clean_name = action_name.strip().lower()

    for rule_key, rule_data in rules.items():
        # Check play conditioning if specified
        if "play" in rule_data and play_type:
            if rule_data["play"].lower() not in play_type.lower():
                continue

        aliases = [a.lower() for a in rule_data.get("aliases", [])]
        if clean_name in aliases or clean_name == rule_key.lower():
            return rule_data

    return None
