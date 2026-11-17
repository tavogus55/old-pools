import json
import math
import os
import shlex
import sys


def load_config(filename, profile, selection=None):
    with open(filename, encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or set(config) - {"experiments", "tuning"}:
        raise ValueError("expected independent experiments and tuning JSON objects")
    settings = config[profile]
    if profile == "experiments":
        experiments = settings if isinstance(settings, list) else [settings]
        if not experiments:
            raise ValueError("experiments must contain at least one experiment")
        commands = [settings_commands(item, profile) for item in experiments]
        names = [item["experiment_name"] for item in experiments]
        if len(set(names)) != len(names):
            raise ValueError("experiments must have unique experiment_name values")
        if selection == "count":
            return "EXPERIMENT_COUNT={}".format(len(experiments))
        index = 0 if not selection else int(selection)
        if not 0 <= index < len(experiments):
            raise ValueError("experiment index out of range: {}".format(index))
        return commands[index]
    return settings_commands(settings, profile)


def settings_commands(settings, profile):
    if not isinstance(settings, dict):
        raise ValueError("each {} entry must be a JSON object".format(profile))
    strings = {"experiment_name", "conda_env", "conda_exe", "python_bin", "mp_layer"}
    integers = {"epochs", "folds", "early_stop", "hidden", "batch_size"}
    numbers = {"learning_rate", "weight_decay", "dropout", "pooling_ratio"}
    flags = {"use_cuda"}
    arrays = {"datasets", "models", "seeds"}
    if profile == "tuning":
        flags.add("quick_test")
        arrays.update({"learning_rates", "weight_decays", "dropouts", "batch_sizes"})
    expected = strings | integers | numbers | flags | arrays
    required = expected if profile == "tuning" else {
        "experiment_name", "conda_env", "conda_exe", "python_bin", "datasets", "models"
    }
    if required - settings.keys() or settings.keys() - expected:
        raise ValueError("missing settings: {}; unknown settings: {}".format(
            ", ".join(sorted(required - settings.keys())),
            ", ".join(sorted(settings.keys() - expected))))
    commands = []
    for key, value in settings.items():
        if key in strings:
            valid = isinstance(value, str) and not any(character in value for character in "\r\n\0")
            if key in {"experiment_name", "conda_env", "mp_layer"}:
                valid = valid and bool(value)
            if key == "mp_layer":
                valid = valid and value in {"gcn", "graphconv"}
        elif key in integers:
            valid = type(value) is int and value > 0
            if key == "folds":
                valid = valid and value >= 2
        elif key in numbers:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            if key == "learning_rate":
                valid = valid and value > 0
            if key in {"dropout", "pooling_ratio"}:
                valid = valid and value <= 1
        elif key in flags:
            valid = type(value) is bool
        else:
            valid = isinstance(value, list)
            if valid and key in {"datasets", "models"}:
                valid = bool(value) and all(isinstance(item, str) and item and not any(character in item for character in ",\t\r\n\0") for item in value)
            elif valid and key in {"seeds", "batch_sizes"}:
                valid = bool(value) and all(type(item) is int and item >= (0 if key == "seeds" else 1) for item in value)
            elif valid:
                valid = bool(value) and all(type(item) in (int, float) and math.isfinite(item) and item >= 0 for item in value)
                if key == "learning_rates":
                    valid = valid and all(item > 0 for item in value)
                if key == "dropouts":
                    valid = valid and all(item <= 1 for item in value)
        if not valid:
            raise ValueError("invalid value for {}: {!r}".format(key, value))
        variable = key.upper()
        if key == "experiment_name":
            variable = "EXP_NAME" if profile == "experiments" else "EXPERIMENT_NAME"
        if key in {"learning_rates", "weight_decays", "dropouts", "batch_sizes"}:
            commands.append("{}=({})".format(key, " ".join(shlex.quote(str(item)) for item in value)))
            continue
        if key in flags:
            value = "1" if value else "0"
        elif key in arrays:
            value = ",".join(str(item) for item in value)
        value = os.environ.get(variable) or str(value)
        commands.append("export {}={}".format(variable, shlex.quote(value)))
    return "\n".join(commands)


if __name__ == "__main__":
    try:
        print(load_config(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
    except (OSError, ValueError, TypeError, KeyError) as error:
        sys.exit("Could not load runner config: {}".format(error))
