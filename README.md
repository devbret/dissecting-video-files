# Dissecting Video Files

An audiovisual remixing pipeline that analyzes video and audio independently, aligns both formats in time and recombines them into a new video driven by audio characteristics. Video files are processed frame by frame to extract visual measurements. These metrics are saved as a JSON file, while a separate script extracts the original video into frame images.

Audio analysis data is aligned to a given frame rate and merged into a new media timeline. During remixing, each audio moment selects a visually appropriate frame based on scoring logic. The selected frames are reordered, assembled into a new sequence and rendered into an MP4.
