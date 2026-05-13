#!/usr/bin/env bash
# scripts/push_to_server.sh
# 一键同步：本地 commit + push 远端 + 服务器 pull。
#
# 用法：
#   bash scripts/push_to_server.sh "your commit message"
#   bash scripts/push_to_server.sh --no-pull "msg"   # 仅 push，不在服务器执行 pull
#   bash scripts/push_to_server.sh --dry-run "msg"   # 仅展示将执行的命令
#
# 通过环境变量配置（推荐写到 ~/.zshrc 或 ~/.bashrc）：
#   XN_SERVER_HOST    SSH 目标（user@host 或 ~/.ssh/config 中的别名）
#   XN_SERVER_PATH    服务器上的项目目录（例：/share/home/sera_hujun/xiamen-nowcast）
#   XN_SSH_OPTS       额外 SSH 选项（可选）

set -euo pipefail

# === 默认配置（请修改为你的真实值，或用环境变量覆盖）=========================
SERVER_HOST="${XN_SERVER_HOST:-CHANGE_ME@CHANGE_ME}"
SERVER_PATH="${XN_SERVER_PATH:-/share/home/sera_hujun/xiamen-nowcast}"
SSH_OPTS="${XN_SSH_OPTS:-}"

# === 命令行参数 ==============================================================
DO_REMOTE_PULL=1
DRY_RUN=0
COMMIT_MSG=""

for arg in "$@"; do
    case "$arg" in
        --no-pull)  DO_REMOTE_PULL=0 ;;
        --dry-run)  DRY_RUN=1 ;;
        --help|-h)
            sed -n '1,20p' "$0"
            exit 0
            ;;
        *)
            if [[ -z "$COMMIT_MSG" ]]; then
                COMMIT_MSG="$arg"
            else
                echo "[push] 忽略多余参数: $arg" >&2
            fi
            ;;
    esac
done

if [[ -z "$COMMIT_MSG" ]]; then
    COMMIT_MSG="sync: $(date '+%Y-%m-%d %H:%M:%S')"
fi

if [[ "$DO_REMOTE_PULL" == "1" && "$SERVER_HOST" == "CHANGE_ME@CHANGE_ME" ]]; then
    echo "[push] 错误：XN_SERVER_HOST 未配置。" >&2
    echo "       请：export XN_SERVER_HOST=user@host  或编辑本脚本顶部默认值。" >&2
    echo "       临时只想 push、不想在服务器 pull，可加 --no-pull。" >&2
    exit 2
fi

# === 准备 ====================================================================
# 兼容首次提交（HEAD 尚不存在）的场景
BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || git config --get init.defaultBranch || echo main)"
echo "[push] 当前分支: $BRANCH"
echo "[push] 服务器  : $SERVER_HOST:$SERVER_PATH (pull=$DO_REMOTE_PULL)"
echo "[push] 提交信息: $COMMIT_MSG"

run() {
    echo "[push] \$ $*"
    if [[ "$DRY_RUN" == "0" ]]; then
        eval "$@"
    fi
}

# === 1) 本地暂存与提交 =======================================================
if [[ -n "$(git status --porcelain)" ]]; then
    run "git add -A"
    run "git commit -m \"$COMMIT_MSG\""
else
    echo "[push] 工作区干净，跳过 commit"
fi

# === 2) 推送到远端 ===========================================================
run "git push origin '$BRANCH'"

# === 3) 服务器侧拉取 =========================================================
if [[ "$DO_REMOTE_PULL" == "1" ]]; then
    REMOTE_CMD="set -e; cd '$SERVER_PATH' && git fetch --all --prune && git checkout '$BRANCH' && git pull --rebase --autostash"
    run "ssh $SSH_OPTS '$SERVER_HOST' \"$REMOTE_CMD\""
fi

echo "[push] 同步完成"
