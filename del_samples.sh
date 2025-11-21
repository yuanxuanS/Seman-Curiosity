# 定义实验名称变量
EXP_NAME="$1"

# --- 清理 dump 目录 ---
TARGET_DIR_1="/home/wpp/Seman-Curiosity/exps/dump/$EXP_NAME/episodes/"
echo "正在删除目录: $TARGET_DIR_1"
rm -rf "$TARGET_DIR_1"

TARGET_DIR_2="/home/wpp/Seman-Curiosity/exps/dump/$EXP_NAME/episodes_data/"
echo "正在删除目录: $TARGET_DIR_2"
rm -rf "$TARGET_DIR_2"

# --- 清理 imgs 目录 ---
TARGET_DIR_3="/home/wpp/Seman-Curiosity/imgs/$EXP_NAME/"
echo "正在删除目录: $TARGET_DIR_3"
rm -rf "$TARGET_DIR_3"

echo "清理完成。"