# P.835 Speech Enhancement Listening Test

A self-contained web listening test that rates each speech sample on the three
ITU-T **P.835** dimensions: speech signal distortion (SIG), background-noise
intrusiveness (BAK), and overall quality (OVRL). Built for a real-time MRI
speech enhancement study comparing 4 systems (3 models + original) across 2 base
forms (raw, post-DSP) = **8 conditions**.

## Files

| File | What it is |
|------|------------|
| `index.html` | The survey. Open in a browser to run it. |
| `generate_stimuli.py` | Regenerates the opaque IDs, mapping, and sample list. |
| `mapping.csv` | **Private key**: maps each opaque `stim_XXXX` id to system / base / utterance. Use this at analysis time. Do **not** publish it in the survey repo. |
| `audioSamples.js.txt` / `.json` | The generated stimulus list (already injected into `index.html`). |
| `audio/` | Put your `.wav` files here, named exactly as in `mapping.csv` (`stim_0001.wav`, etc.) plus `practice_01.wav` / `practice_02.wav`. |

## 1. Prepare your audio

1. Render one `.wav` per condition cell. There are `4 systems x 2 bases x 12
   utterances = 96` stimuli by default.
2. **Loudness-normalise every file to the same target** (e.g. -23 LUFS) before
   uploading. Level differences bias quality ratings more than almost anything.
   Example with ffmpeg:
   ```bash
   ffmpeg -i in.wav -af loudnorm=I=-23:LRA=7:TP=-2 -ar 16000 out.wav
   ```
3. Rename each file to the opaque id from `mapping.csv` (the `filename` column)
   and drop it in `audio/`. The opaque names keep the condition hidden from
   listeners and from anyone viewing page source.
4. Add 1-2 practice clips as `audio/practice_01.wav` and `audio/practice_02.wav`.

To change the number of utterances/systems/bases, edit the top of
`generate_stimuli.py`, re-run `python3 generate_stimuli.py`, then re-inject the
array (the script writes `audioSamples.js.txt`; paste it between the
`AUDIO SAMPLES` markers in `index.html`).

## 2. Configure the survey

Edit the CONFIGURATION block near the bottom of `index.html`:

- `TASK_ID` — label stored with every response.
- `UTTERANCES_PER_SESSION` — per-listener load. With 96 stimuli, leaving this at
  `4` means each listener rates all 8 conditions for 4 utterances = 32 samples
  (~20-25 min). Distribute listeners across blocks with the URL parameter
  `?block=1`, `?block=2`, `?block=3` so every utterance gets covered. Set to `0`
  to show all 96 to everyone.
- `WEBHOOK_URL` — where results are sent (see step 4). Leave `""` for
  download-only.

## 3. Host on GitHub Pages

Because the audio lives in the **same** repo as `index.html`, everything is
same-origin and you do not need to configure CORS.

1. Create a new GitHub repo (e.g. `p835-study`) and add these files plus your
   populated `audio/` folder. **Leave `mapping.csv` out of this repo** (or put it
   in a separate private repo) so listeners can't decode conditions.
2. Push to the `main` branch.
3. In the repo: **Settings -> Pages -> Build and deployment**. Set
   **Source = Deploy from a branch**, **Branch = main**, **folder = / (root)**,
   and Save.
4. After a minute your test is live at
   `https://<your-username>.github.io/p835-study/` (append `?block=1` etc.).
5. Test it yourself end-to-end before recruiting.

Notes: GitHub Pages serves over HTTPS, which audio playback requires. Large WAVs
load fine, but if total audio is very large consider 16 kHz mono WAV or FLAC to
cut size. The repo file-size limit is 100 MB per file (not an issue for speech).

## 4. Getting results by email

The survey always downloads a JSON copy to the listener's computer. To also
receive each submission by email, point `WEBHOOK_URL` at one of these (no server
needed):

### Option A — Formspree (quickest)
1. Sign up at formspree.io, create a form, copy its endpoint
   (`https://formspree.io/f/abcxyz`).
2. Set `WEBHOOK_URL = "https://formspree.io/f/abcxyz"` in `index.html`.
3. Each completed survey is emailed to you as JSON. Free tier covers small
   studies; check current submission limits.

### Option B — Google Apps Script (logs to a Sheet *and* emails)
1. Create a Google Sheet. Extensions -> Apps Script, paste:
   ```javascript
   function doPost(e) {
     var data = JSON.parse(e.postData.contents);
     var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
     sheet.appendRow([new Date(), data.completionCode, JSON.stringify(data.responses)]);
     MailApp.sendEmail("foley.sean3@gmail.com",
       "New P.835 response " + data.completionCode,
       JSON.stringify(data.responses, null, 2));
     return ContentService.createTextOutput(JSON.stringify({ok:true}))
       .setMimeType(ContentService.MimeType.JSON);
   }
   ```
2. Deploy -> New deployment -> type **Web app**, execute as **you**, access
   **Anyone**. Copy the `/exec` URL into `WEBHOOK_URL`.
3. Note: a cross-origin POST to Apps Script can hit CORS; if the browser blocks
   it the local JSON download still works, and the listener is told to email the
   file. Formspree is the more reliable email path.

If you only want JSON files (no email), leave `WEBHOOK_URL = ""` and collect the
downloaded files (e.g. via your crowdsourcing platform's file upload).

## 5. Analysis

Each response JSON keys ratings by opaque id:
```json
{ "stim_0007": { "SIG": 4, "BAK": 2, "OVRL": 3 }, ... }
```
Join on `id` against `mapping.csv` to recover system / base / utterance, then
compute mean SIG/BAK/OVRL per condition with confidence intervals. Aim for
**>= 15-20 ratings per stimulus** for stable means; recruit enough listeners
across the blocks to reach that.

## Quality-control tips already supported
- Three scales presented in fixed P.835 order (signal -> background -> overall).
- All three required before advancing; no going back.
- Per-listener shuffle with same-utterance spacing.
- Practice/familiarisation block (important for unusual rtMRI audio).
- A final attention-check question.
- Consider seeding 1-2 obvious "gold" items (e.g. clean original) per block and
  rejecting listeners who rate them implausibly low.
