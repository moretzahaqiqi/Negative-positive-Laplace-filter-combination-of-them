import cv2
import numpy as np

# خواندن تصویر سیاه و سفید
image_path = 'image.jpg'
image = cv2.imread(image_path, 0)

if image is None:
    print("❌ خطا: تصویر پیدا نشد!")
    exit()

# تبدیل float برای محاسبات دقیق
image_float = image.astype(np.float32)

# فیلترهای لاپلاس
kernel_positive = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
kernel_negative = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

# اعمال فیلترها
laplacian_positive = cv2.filter2D(image_float, -1, kernel_positive)
laplacian_negative = cv2.filter2D(image_float, -1, kernel_negative)

# ترکیب‌های مختلف
combo_pos = image_float + laplacian_positive
combo_neg = image_float + laplacian_negative
combo_all = image_float + laplacian_positive - laplacian_negative

# نرمال‌سازی
def normalize(img):
    return np.clip(img, 0, 255).astype(np.uint8)

# تبدیل همه به uint8
original = image
positive = normalize(laplacian_positive)
negative = normalize(laplacian_negative)
combo_p = normalize(combo_pos)
combo_n = normalize(combo_neg)
combo_a = normalize(combo_all)

# تبدیل همه به تصویر BGR برای نمایش رنگی
def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

imgs = [
    to_bgr(original),
    to_bgr(positive),
    to_bgr(negative),
    to_bgr(combo_p),
    to_bgr(combo_n),
    to_bgr(combo_a)
]

labels = [
    "Original",
    "Laplacian Positive",
    "Laplacian Negative",
    "Original + Positive",
    "Original + Negative",
    "Original + Pos + Neg"
]

# اندازه‌ها
height, width = original.shape
label_height = 40
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
thickness = 1
text_color = (0, 255, 0)

# تابع اضافه کردن متن بالای تصویر
def add_label(img, text):
    labeled_img = np.zeros((img.shape[0] + label_height, img.shape[1], 3), dtype=np.uint8)
    labeled_img[label_height:] = img
    cv2.putText(labeled_img, text, (10, 25), font, font_scale, text_color, thickness)
    return labeled_img

# افزودن برچسب به همه تصاویر
labeled_images = [add_label(imgs[i], labels[i]) for i in range(len(imgs))]

# ترکیب تمام تصاویر در یک خط افقی
combined = np.hstack(labeled_images)

# ذخیره و نمایش
cv2.imshow("All Combinations", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

output_path = "all_combinations.jpg"
cv2.imwrite(output_path, combined)
print(f"✅ تصویر تمام ترکیب‌ها ذخیره شد: {output_path}")