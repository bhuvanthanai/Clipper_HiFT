# Auto clipping — research and chosen approach

Goal: pick the best 15–60 s sections of a long video (mostly stream VODs) without hand-picking `--start` / `--end`, on a normal CPU, with no paid APIs.

## Approaches compared

- YouTube "most replayed" heatmap (yt-dlp `heatmap`, 100 points): free and the best signal when present; missing on many videos.
- Live-chat replay rate (yt-dlp `live_chat` subtitles): strong for stream VODs (LIGHTOR, MINT); needs ~8 s reaction-lag correction.
- Audio loudness peaks (RMS vs 60 s rolling baseline): cheap and always available; catches shouting, laughter, hype.
- Transcript cues (words/s, exclamation and laughter tokens): free; weak alone, good tie-breaker.
- Motion / scene change (frame diffs on a 64x36 decode of the 360p copy): seconds per hour; helps gameplay action.
- Audio event model (YAMNet laughter/applause): better laughter detection; adds a ~15 MB model and a TF runtime.
- LLM transcript scoring: best semantic quality; hosted is paid, local 3–8 B models need several GB RAM and are slow on CPU.
- Multimodal deep models (Ringer et al., MINT): need per-channel training data; out of scope.

## Chosen approach

- Weighted fusion of the cheap signals: heatmap 0.35, chat 0.25, audio 0.2, speech 0.1, motion 0.1.
- Weights renormalise over the signals that are present, so a local file with only audio + transcript still works.
- Each signal is per-second, robust-z-scored (median/MAD) and clipped to 0..1, so no single spike dominates.
- Candidate windows start at transcript segment starts and end at sentence ends or ≥ 0.5 s pauses.
- Window score = 0.6 × mean + 0.4 × mean of its top 5 seconds (rewards sustained interest plus a payoff moment).
- Greedy non-overlapping pick of the top N windows; each gets a human-readable `reason`.

Why: it runs in seconds on CPU with no models, uses the strongest free signals (heatmap, chat) when they exist, and degrades gracefully when they don't. LLM scoring can be added later as an optional re-ranker over the top candidates.

## Findings from the sample

- The sample VOD (`o0UeucVxYCA`, 3 h live stream) has no `heatmap` in yt-dlp metadata, fetched 2026-11-19, so chat and audio carry stream VODs.
- yt-dlp's YouTube extractor does expose `heatmap` (100 points) for videos that have it.

## Sources

- [AI-Youtube-Shorts-Generator](https://github.com/Anil-matcha/AI-Youtube-Shorts-Generator)
- [Prompt2Clip (audio loudness + YAMNet + LLM)](https://github.com/Jit-Roy/Prompt2Clip)
- [Sieve: finding highlights in long-form video](https://sievedata.com/blog/generate-video-highlights-long-form-content-podcasts)
- [LIGHTOR (chat bursts, ICDE 2020)](https://www.microsoft.com/en-us/research/uploads/prod/2020/02/icde20lightor.pdf)
- [MINT, EPJ Data Science 2021](https://blogs.biomedcentral.com/on-physicalsciences/2021/09/02/epic-moments-live-streaming/)
- [Ringer, livestream highlight detection thesis](https://etheses.whiterose.ac.uk/32406/)
- [Fu et al., highlight prediction from chat reactions](https://arxiv.org/abs/1707.08559)
- [Most-replayed data location (Stack Overflow)](https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/questions/72610552/most-replayed-data-of-youtube-video-via-api)
