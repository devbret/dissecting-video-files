import json
import bisect
import numpy as np

AUDIO_ANALYSIS_JSON = "path/to/your/audio_analysis_enhanced.json"
OUTPUT_JSON = "frame_aligned_audio.json"

CHOSEN_FPS = 60

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found -> {path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON decode error -> {e}")
        return {}

def get_audio_duration(audio_data):
    if not audio_data:
        return 0.0

    first_audio_key = next(iter(audio_data.keys()))
    audio_dict = audio_data[first_audio_key]

    max_time = 0.0

    def scan_time_value_list(lst):
        return max(d.get("time", 0.0) for d in lst) if lst else 0.0

    def scan_sub_metrics(sub_dict_list):
        local_max = 0.0
        for sub_dict in sub_dict_list:
            for _, tv_list in sub_dict.items():
                local_max = max(local_max, scan_time_value_list(tv_list))
        return local_max

    for key, value in audio_dict.items():
        if isinstance(value, (int, float)):
            continue
        elif isinstance(value, list):
            if all(isinstance(x, dict) and "time" in x and "value" in x for x in value):
                max_time = max(max_time, scan_time_value_list(value))
            else:
                if all(isinstance(x, dict) for x in value):
                    local_max = scan_sub_metrics(value)
                    max_time = max(max_time, local_max)
        else:
            pass

    return max_time

def flatten_simple_list_of_time_value(data_list):
    sorted_list = sorted(data_list, key=lambda x: x["time"])
    times = [d["time"] for d in sorted_list]
    values = [d["value"] for d in sorted_list]
    return (times, values)

def flatten_multi_channel_dicts(data_list):
    combined_subkeys = {}

    for item in data_list:
        for subkey, tv_list in item.items():
            if subkey not in combined_subkeys:
                combined_subkeys[subkey] = []
            combined_subkeys[subkey].extend(tv_list)

    sub_lookup = {}
    for subkey, tv_list in combined_subkeys.items():
        sub_lookup[subkey] = flatten_simple_list_of_time_value(tv_list)
    return sub_lookup

def nearest_value(times_list, values_list, target_time):
    if not times_list or not values_list:
        return None
    pos = bisect.bisect_left(times_list, target_time)
    if pos == 0:
        return values_list[0]
    if pos == len(times_list):
        return values_list[-1]
    before_time = times_list[pos - 1]
    after_time = times_list[pos]
    if abs(after_time - target_time) < abs(before_time - target_time):
        return values_list[pos]
    else:
        return values_list[pos - 1]

def build_audio_lookup(audio_dict):
    lookup = {}

    for key, value in audio_dict.items():
        if isinstance(value, (int, float)):
            lookup[key] = ("single_value", float(value))

        elif isinstance(value, list):
            if all(isinstance(x, dict) and "time" in x and "value" in x for x in value):
                tv_pair = flatten_simple_list_of_time_value(value)
                lookup[key] = ("time_value", tv_pair)
            else:
                if all(isinstance(x, dict) for x in value):
                    sub_lookup = flatten_multi_channel_dicts(value)
                    lookup[key] = ("multi_channel", sub_lookup)
                else:
                    lookup[key] = ("unhandled", value)
        else:
            lookup[key] = ("unhandled", value)

    return lookup

def get_nearest_metric_value(struct_type, struct_data, frame_time):
    if struct_type == "single_value":
        return struct_data

    elif struct_type == "time_value":
        times_list, values_list = struct_data
        return nearest_value(times_list, values_list, frame_time)

    elif struct_type == "multi_channel":
        result = {}
        for subkey, (times_list, values_list) in struct_data.items():
            val = nearest_value(times_list, values_list, frame_time)
            result[subkey] = val
        return result

    elif struct_type == "unhandled":
        return None

    return None

def main():
    audio_data = load_json(AUDIO_ANALYSIS_JSON)
    if not audio_data:
        print("No audio data found.")
        return

    first_audio_key = next(iter(audio_data.keys()))
    audio_dict = audio_data[first_audio_key]

    audio_duration = get_audio_duration(audio_data)
    print(f"Audio file: {first_audio_key}, Duration: {audio_duration:.2f} seconds")

    audio_lookup = build_audio_lookup(audio_dict)

    frame_times = np.arange(0, audio_duration, 1.0 / CHOSEN_FPS)

    aligned_audio_data = []
    for t in frame_times:
        frame_info = {"time": float(t)}
        for metric_name, (struct_type, struct_data) in audio_lookup.items():
            val = get_nearest_metric_value(struct_type, struct_data, t)
            frame_info[metric_name] = val
        aligned_audio_data.append(frame_info)

    output_data = {
        "audio_file": first_audio_key,
        "audio_duration": audio_duration,
        "fps": CHOSEN_FPS,
        "aligned_audio_data": aligned_audio_data
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Success! Wrote {len(aligned_audio_data)} entries to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
