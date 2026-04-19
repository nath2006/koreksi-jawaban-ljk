import cv2
import numpy as np

img = cv2.imread("test_ljk.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray_eq = clahe.apply(gray)
blurred = cv2.GaussianBlur(gray_eq, (5,5), 0)

thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 8)
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if 12 < w < 70 and 12 < h < 70 and 0.4 < (w/h) < 2.5:
        boxes.append((x,y,w,h))

med_w = np.median([b[2] for b in boxes])
med_h = np.median([b[3] for b in boxes])
filtered = [b for b in boxes if 0.5*med_w < b[2] < 1.8*med_w and 0.5*med_h < b[3] < 1.8*med_h]

def remove_overlaps(boxes):
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    keep = []
    for box in boxes:
        cx1, cy1 = box[0]+box[2]//2, box[1]+box[3]//2
        is_dup = False
        for kept in keep:
            cx2, cy2 = kept[0]+kept[2]//2, kept[1]+kept[3]//2
            if abs(cx1-cx2) < min(box[2],kept[2])*0.5 and abs(cy1-cy2) < min(box[3],kept[3])*0.5:
                is_dup = True
                break
        if not is_dup:
            keep.append(box)
    return keep

unique = remove_overlaps(filtered)

# Show ALL boxes in Col 5 area (x > 850)
col5_all = sorted([b for b in unique if b[0]+b[2]//2 > 850], key=lambda b: (b[1], b[0]))
print(f"All boxes with X > 850: {len(col5_all)}")
print(f"These should be exactly 50 (5 choices x 10 rows)")

# Group by Y
col5_all.sort(key=lambda b: b[1]+b[3]//2)
rows = []
current_row = [col5_all[0]]
for i in range(1, len(col5_all)):
    cy_prev = current_row[-1][1]+current_row[-1][3]//2
    cy_curr = col5_all[i][1]+col5_all[i][3]//2
    if abs(cy_curr - cy_prev) < med_h * 0.7:
        current_row.append(col5_all[i])
    else:
        rows.append(current_row)
        current_row = [col5_all[i]]
rows.append(current_row)

print(f"\nRows found: {len(rows)}")
for i, row in enumerate(rows):
    row.sort(key=lambda b: b[0])
    print(f"\nRow {i+1} ({len(row)} boxes):")
    for j, (x,y,w,h) in enumerate(row):
        cx = x+w//2
        roi = gray[y+int(h*0.25):y+h-int(h*0.25), x+int(w*0.25):x+w-int(w*0.25)]
        val = np.mean(roi) if roi.size > 0 else 255
        dark = " <-- DARK" if val < 80 else ""
        print(f"  Box {j}: x={x:4d}-{x+w:4d} (cx={cx:4d}) y={y:3d}-{y+h:3d} ({w}x{h}) intensity={val:5.1f}{dark}")

# Also check what's just to the LEFT of col 5 (the gap area)
print(f"\n--- Checking gap between Col4 and Col5 ---")
gap_boxes = sorted([b for b in unique if 820 < b[0]+b[2]//2 < 890], key=lambda b: (b[1], b[0]))
print(f"Boxes in gap area (x=820-890): {len(gap_boxes)}")
for b in gap_boxes:
    x,y,w,h = b
    print(f"  x={x}-{x+w} y={y}-{y+h} ({w}x{h})")
