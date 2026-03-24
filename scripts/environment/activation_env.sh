# 만약 파이썬 환경 활성화 되어있다면 deactivate
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Deactivating current virtual environment: $VIRTUAL_ENV"
    deactivate
else
    echo "No active virtual environment found. Proceeding without deactivation."
fi

# 환경변수 목록 가져오기
# ENV_VARS=$(compgen -v)
# echo "Current environment variables before activation:"
# for var in $ENV_VARS; do
#     echo "  $var=${!var}"
# done

# 원하는 가상환경 활성화 (예: venv라면 경로만 수정해서 사용)
source "$HOME/venvs/nnunet_torch280/bin/activate"
#source "$HOME/venvs/tf220/bin/activate"
echo "Activated virtual environment: $(basename "$VIRTUAL_ENV")"

