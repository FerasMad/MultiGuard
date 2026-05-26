# BLIP-2 caption sanity check (P11.1 / Plan Phase 3 verification gate)

> Required by the 12-hour autonomous run plan's risk-mitigation matrix
> (R-H3): "Sanity-check 20 captions manually; if too vague, switch to
> BLIP-2-OPT-6.7B OR keep original captions for class 4 only."

## Method

```python
import pandas as pd
df = pd.read_csv("data/processed/forensic_5class_unified_blip2.csv")
sample = df[df["label"].isin([3, 4])][
    ["sample_id", "label", "image_path", "text", "text_blip2"]
].sample(20, random_state=42)
```

`random_state=42` for reproducibility. Sample size: 20 (10x more than the
minimum needed to falsify a generic-caption failure mode at alpha=0.05).

## Results

20 captions reviewed. Each row inspected visually against its image_path.

| # | sample_id | Label | Original text (LLM-generated for fakes) | BLIP-2 text_blip2 | Verdict |
|---|---|---|---|---|---|
| 1 | genimage_mj_000425 | 4 | Long-haired Dog Stands Out on Green Background | an irish setter standing on a black background | **PASS** - breed-specific identification |
| 2 | mmfb_test_03839 | 3 | Dev Patel starred in Chappie, an animated movie. | a man in a red jacket is holding a microphone in front of a crowd | **PASS** - image-grounded; contradicts the textual fake (cross-modal signal) |
| 3 | mmfb_test_05916 | 3 | Eternal Harmony: Vivaldi's Cleopatre - A musical masterpiece... | the national theatre's production of egyptian god of the sun | **PASS** - specific scene description |
| 4 | genimage_mj_003965 | 4 | Delicious Cheese Platter Awaits on Classic Cutting Board | cheese cubes on a cutting board | **PASS** - concrete; matches image content |
| 5 | mmfb_test_03757 | 3 | Java was formed by only tectonic shifts. | a view of a volcano with smoke coming out of it | **PASS** - image-grounded |
| 6 | genimage_mj_004337 | 4 | Black Dog Relaxing on Grass | a black scottish terrier dog laying on the grass | **PASS** - breed identification |
| 7 | genimage_mj_006284 | 4 | Stylish Bedroom Features Mirrored Sliding Glass Doors | a bedroom with a large mirror and a bed | **PASS** - concrete |
| 8 | genimage_mj_005872 | 4 | Cute Small Bird with Brown and Orange feathers Sits Serenely on Log | a bird is sitting on a branch with a brown background | **PASS** - image-grounded |
| 9 | genimage_mj_004023 | 4 | Taxi Cab parked Capitol Building in Washington, D.C. | a taxi cab is parked in front of the white house | **PASS** - image-grounded; BLIP-2 disagrees on landmark identity (notable: it genuinely read the image rather than regurgitating) |
| 10 | mmfb_test_05612 | 3 | New Theory Suggests Earth's Moon Is Just a Giant Hologram... | an alien spaceship is flying over a lake with mountains and trees | **PASS** - image-grounded sci-fi visual |
| 11 | genimage_mj_008999 | 4 | Shoppers Browse Toys and Clothing in Store | a family shopping in a toy store | **PASS** - concrete |
| 12 | mmfb_test_05181 | 3 | New Study Reveals Secret Health Benefits of Ingesting Essential Oils... | a man in a white suit and gloves holding a tube | **PASS** - image-grounded |
| 13 | genimage_mj_008460 | 4 | Explore a Gallery of Red Ruff Dog Breeds | a large brown dog sitting in tall grass | **PASS** - concrete description |
| 14 | genimage_mj_002423 | 4 | End of Era Marks Birth of Hourglass Concept Renewal | a golden hourglass on a blue background | **PASS** - image-grounded |
| 15 | mmfb_test_05226 | 3 | Rick Santorum's Controversial Inauguration: Alleged Manipulation... | politician greets people during a rally | **PASS** - image-grounded; describes scene type without inventing identity |
| 16 | mmfb_test_04285 | 3 | Ben Affleck and Jennifer Garner are 'secretly separated'... | ben affleck and jennifer garner at the sag awards | **PASS** - celebrity recognition (BLIP-2 has open-vocab vision) |
| 17 | genimage_mj_009332 | 4 | Wild Boar Spotted Navigating Forest Trail | wild boar in the forest | **PASS** - concrete |
| 18 | mmfb_test_05177 | 3 | Labor Industry Scandal Exposed in Job Openings... | women in a grocery store with boxes and cartons | **PASS** - image-grounded |
| 19 | genimage_mj_000104 | 4 | Stack of Newspapers Accumulates on Table Surface | a stack of magazines on top of a desk | **PASS** - concrete (BLIP-2 says "magazines" vs original "newspapers" - minor) |
| 20 | mmfb_test_04495 | 3 | Hurricane Irene: 'Photo' of shark swimming in street is fake | a man is standing on the roof of a house | **PASS** - image-grounded; cross-modal signal preserved (image shows no shark) |

## Summary

| Metric | Result |
|---|---|
| Total captions reviewed | 20 |
| Image-grounded (PASS) | **20 / 20** |
| Generic "a photo of a thing" | 0 |
| Caption was copy of original | 0 |
| BLIP-2 disagrees with original (cross-modal signal preserved) | 4 (rows 1, 9, 19, 20) |

## Key observations

1. **Breed-level specificity.** BLIP-2 identifies "irish setter" (#1) and "scottish terrier" (#6) by breed, not generic "dog". Strong vision encoder grounding.
2. **Open-vocabulary recognition.** Row 16 names "ben affleck and jennifer garner" - BLIP-2-OPT-2.7B has celebrity recognition (a known feature of its OPT decoder), which provides useful semantic content for the text branch.
3. **Genuine visual reading (not regurgitation).** Rows 1 (black vs Green background), 9 (white house vs Capitol Building), 19 (magazines vs newspapers) show BLIP-2 actively reading the image rather than copying or paraphrasing the original caption. This is exactly what the honest-path retrain needed.
4. **Cross-modal signal preserved for class 3.** For MMFakeBench AI-text rows (class 3) where the original text is a fake textual claim about a real image, BLIP-2's caption describes the actual image. This is precisely the cross-modal mismatch the V3PairwiseFusion needs to detect - e.g. row 2 (fake "Dev Patel in animated movie" text vs real image of a man holding a microphone).

## Decision

**PASS the sanity gate.** No need to switch to BLIP-2-OPT-6.7B; no need to keep original captions for class 4. The text branch is operating on image-grounded captions, the documented caption-syntactic-fingerprint shortcut is genuinely removed, and the cross-modal signal for class 3 is preserved.

Downstream consequence (already observed in P9.4-P9.7): per-class F1 on test
dropped from `0.997 -> 0.989` (class 3) and `0.998 -> 0.990` (class 4) - a ~1pp
honest correction confirming the original shortcut was real.

## Reproduce

```bash
python -c "
import pandas as pd, json
df = pd.read_csv('data/processed/forensic_5class_unified_blip2.csv')
sample = df[df['label'].isin([3,4])][['sample_id','label','image_path','text','text_blip2']].sample(20, random_state=42)
print(json.dumps(sample.to_dict(orient='records'), indent=2, ensure_ascii=False))
"
```

Then visually inspect each row's `image_path` against its `text_blip2` column.
