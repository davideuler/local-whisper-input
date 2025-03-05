import os
import sys

from dotenv import load_dotenv

load_dotenv()

from src.audio.recorder import AudioRecorder
from src.keyboard.listener import KeyboardManager, check_accessibility_permissions
from src.transcription.whisper import WhisperProcessor
from src.utils.logger import logger
from src.transcription.senseVoiceSmall import SenseVoiceSmallProcessor


# 导入本地模型处理器
LOCAL_SENSEVOICE_AVAILABLE = False
LOCAL_WHISPER_AVAILABLE = False

# 尝试导入本地 SenseVoiceSmall 实现（使用 funasr-onnx）
try:
    from src.transcription.localSenseVoiceSmall import LocalSenseVoiceSmallProcessor
    LOCAL_SENSEVOICE_AVAILABLE = True
except ImportError:
    logger.warning("本地 SenseVoiceSmall 模型 (funasr-onnx) 不可用，请安装必要的依赖: pip install -U funasr-onnx")

# 尝试导入本地 Whisper 模型
try:
    from src.transcription.localWhisper import LocalWhisperProcessor
    LOCAL_WHISPER_AVAILABLE = True
except ImportError:
    logger.warning("本地 Whisper 模型不可用，请安装必要的依赖: pip install -U openai-whisper")


def check_microphone_permissions():
    """检查麦克风权限并提供指导"""
    logger.warning("\n=== macOS 麦克风权限检查 ===")
    logger.warning("此应用需要麦克风权限才能进行录音。")
    logger.warning("\n请按照以下步骤授予权限：")
    logger.warning("1. 打开 系统偏好设置")
    logger.warning("2. 点击 隐私与安全性")
    logger.warning("3. 点击左侧的 麦克风")
    logger.warning("4. 点击右下角的锁图标并输入密码")
    logger.warning("5. 在右侧列表中找到 Terminal（或者您使用的终端应用）并勾选")
    logger.warning("\n授权后，请重新运行此程序。")
    logger.warning("===============================\n")

class VoiceAssistant:
    def __init__(self, audio_processor):
        self.audio_recorder = AudioRecorder()
        self.audio_processor = audio_processor
        self.keyboard_manager = KeyboardManager(
            on_record_start=self.start_transcription_recording,
            on_record_stop=self.stop_transcription_recording,
            on_translate_start=self.start_translation_recording,
            on_translate_stop=self.stop_translation_recording,
            on_reset_state=self.reset_state
        )
    
    def start_transcription_recording(self):
        """开始录音（转录模式）"""
        self.audio_recorder.start_recording()
    
    def stop_transcription_recording(self):
        """停止录音并处理（转录模式）"""
        audio = self.audio_recorder.stop_recording()
        if audio == "TOO_SHORT":
            logger.warning("录音时长太短，状态将重置")
            self.keyboard_manager.reset_state()
        elif audio:
            result = self.audio_processor.process_audio(
                audio,
                mode="transcriptions",
                prompt=""
            )
            # 解构返回值
            text, error = result if isinstance(result, tuple) else (result, None)
            self.keyboard_manager.type_text(text, error)
        else:
            logger.error("没有录音数据，状态将重置")
            self.keyboard_manager.reset_state()
    
    def start_translation_recording(self):
        """开始录音（翻译模式）"""
        self.audio_recorder.start_recording()
    
    def stop_translation_recording(self):
        """停止录音并处理（翻译模式）"""
        audio = self.audio_recorder.stop_recording()
        if audio == "TOO_SHORT":
            logger.warning("录音时长太短，状态将重置")
            self.keyboard_manager.reset_state()
        elif audio:
            result = self.audio_processor.process_audio(
                    audio,
                    mode="translations",
                    prompt=""
                )
            text, error = result if isinstance(result, tuple) else (result, None)
            self.keyboard_manager.type_text(text,error)
        else:
            logger.error("没有录音数据，状态将重置")
            self.keyboard_manager.reset_state()

    def reset_state(self):
        """重置状态"""
        self.keyboard_manager.reset_state()
    
    def run(self):
        """运行语音助手"""
        logger.info("=== 语音助手已启动 ===")
        self.keyboard_manager.start_listening()

def main():
    # 判断使用哪种服务平台
    service_platform = os.getenv("SERVICE_PLATFORM", "local")
    use_local = service_platform == "local" or os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
    
    # 检查是否有 API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
    
    # 获取本地模型配置
    local_model = os.getenv("LOCAL_MODEL", "SenseVoiceSmall").lower()
    
    # 如果指定使用本地模型或者没有配置 API key，尝试使用本地模型
    if use_local or (not groq_api_key and not siliconflow_api_key):
        # 根据配置选择不同的本地模型
        if local_model == "sensevoicesmall" and LOCAL_SENSEVOICE_AVAILABLE:
            logger.info("使用本地 SenseVoiceSmall 模型进行语音转录 (funasr-onnx)")
            try:
                audio_processor = LocalSenseVoiceSmallProcessor()
                # 如果成功初始化本地模型，直接使用
                try:
                    assistant = VoiceAssistant(audio_processor)
                    assistant.run()
                    return
                except Exception as e:
                    logger.error(f"运行本地 SenseVoiceSmall 模型 (funasr-onnx) 失败: {e}")
                    logger.error("将尝试使用其他模型或在线服务...")
            except Exception as e:
                logger.error(f"初始化本地 SenseVoiceSmall 模型 (funasr-onnx) 失败: {e}")
                logger.error("将尝试使用其他模型或在线服务...")
        
        # 尝试使用 Whisper 模型
        elif LOCAL_WHISPER_AVAILABLE and local_model in ["whisperbase", "whisperv2large", "whisperv3large"]:
            # 映射模型名称
            whisper_model_map = {
                "whisperbase": "base",
                "whisperv2large": "large-v2",
                "whisperv3large": "large-v3"
            }
            
            model_name = whisper_model_map.get(local_model, "base")
            logger.info(f"使用本地 Whisper 模型进行语音转录 ({model_name})")
            
            try:
                audio_processor = LocalWhisperProcessor(model_name=model_name)
                # 如果成功初始化本地模型，直接使用
                try:
                    assistant = VoiceAssistant(audio_processor)
                    assistant.run()
                    return
                except Exception as e:
                    logger.error(f"运行本地 Whisper 模型 ({model_name}) 失败: {e}")
                    logger.error("将尝试使用在线服务...")
            except Exception as e:
                logger.error(f"初始化本地 Whisper 模型 ({model_name}) 失败: {e}")
                logger.error("将尝试使用在线服务...")
        
        # 如果指定的模型不可用，尝试使用任何可用的本地模型
        elif LOCAL_SENSEVOICE_AVAILABLE:
            logger.warning(f"指定的本地模型 '{local_model}' 不可用，将使用 SenseVoiceSmall 模型")
            try:
                audio_processor = LocalSenseVoiceSmallProcessor()
                # 如果成功初始化本地模型，直接使用
                try:
                    assistant = VoiceAssistant(audio_processor)
                    assistant.run()
                    return
                except Exception as e:
                    logger.error(f"运行本地 SenseVoiceSmall 模型 (funasr-onnx) 失败: {e}")
                    logger.error("将尝试使用在线服务...")
            except Exception as e:
                logger.error(f"初始化本地 SenseVoiceSmall 模型 (funasr-onnx) 失败: {e}")
                logger.error("将尝试使用在线服务...")
        elif LOCAL_WHISPER_AVAILABLE:
            logger.warning(f"指定的本地模型 '{local_model}' 不可用，将使用 Whisper Base 模型")
            try:
                audio_processor = LocalWhisperProcessor(model_name="base")
                # 如果成功初始化本地模型，直接使用
                try:
                    assistant = VoiceAssistant(audio_processor)
                    assistant.run()
                    return
                except Exception as e:
                    logger.error(f"运行本地 Whisper Base 模型失败: {e}")
                    logger.error("将尝试使用在线服务...")
            except Exception as e:
                logger.error(f"初始化本地 Whisper Base 模型失败: {e}")
                logger.error("将尝试使用在线服务...")
        
        if not LOCAL_SENSEVOICE_AVAILABLE and not LOCAL_WHISPER_AVAILABLE:
            logger.warning("本地模型不可用，将尝试使用在线服务")
            use_local = False
    
    # 如果不使用本地模型，则使用在线服务
    if not use_local or (not LOCAL_SENSEVOICE_AVAILABLE and not LOCAL_WHISPER_AVAILABLE):
        if service_platform == "groq" and groq_api_key:
            logger.info("使用 Groq Whisper 服务进行语音转录")
            audio_processor = WhisperProcessor()
        elif (service_platform == "siliconflow" or not service_platform) and siliconflow_api_key:
            logger.info("使用 SiliconFlow 服务进行语音转录")
            audio_processor = SenseVoiceSmallProcessor()
        else:
            available_services = []
            if groq_api_key:
                available_services.append("groq")
            if siliconflow_api_key:
                available_services.append("siliconflow")
            
            if available_services:
                # 使用第一个可用的服务
                service_platform = available_services[0]
                logger.info(f"未指定有效的服务平台，自动选择: {service_platform}")
                if service_platform == "groq":
                    audio_processor = WhisperProcessor()
                else:
                    audio_processor = SenseVoiceSmallProcessor()
            else:
                logger.error("未配置任何 API key，且本地模型不可用")
                logger.error("请配置 GROQ_API_KEY 或 SILICONFLOW_API_KEY 环境变量，或安装本地模型依赖")
                sys.exit(1)
    
    try:
        assistant = VoiceAssistant(audio_processor)
        assistant.run()
    except Exception as e:
        error_msg = str(e)
        if "Input event monitoring will not be possible" in error_msg:
            check_accessibility_permissions()
            sys.exit(1)
        elif "无法访问音频设备" in error_msg:
            check_microphone_permissions()
            sys.exit(1)
        else:
            logger.error(f"发生错误: {error_msg}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    main() 
