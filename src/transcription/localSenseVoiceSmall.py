import os, traceback
import time
import io
import tempfile
import soundfile as sf
from pathlib import Path
from ..utils.logger import logger

class LocalSenseVoiceSmallProcessor:
    """本地 SenseVoiceSmall 转录处理器，不依赖于 API key"""
    
    DEFAULT_MODEL_DIR = "iic/SenseVoiceSmall"
    
    def __init__(self):
        try:
            # 尝试导入必要的库
            from funasr_onnx import SenseVoiceSmall
            from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
            
            self.SenseVoiceSmall = SenseVoiceSmall
            self.rich_transcription_postprocess = rich_transcription_postprocess
            
            # 检查模型目录是否存在，如果不存在则自动下载
            model_dir = os.getenv("LOCAL_SENSEVOICE_MODEL_DIR", self.DEFAULT_MODEL_DIR)
            self.model_dir = model_dir
            
            # 初始化模型
            logger.info(f"正在加载本地 SenseVoiceSmall 模型 ({model_dir})...")
            self.model = self.SenseVoiceSmall(model_dir, batch_size=1, quantize=True)
            logger.info("本地 SenseVoiceSmall 模型加载完成")
            
            
            self.translate_processor = None
            # 如果需要翻译功能，可以在这里初始化翻译处理器
            if os.getenv("ENABLE_LOCAL_TRANSLATION", "false").lower() == "true":
                try:
                    from ..llm.translate import TranslateProcessor
                    self.translate_processor = TranslateProcessor()
                    logger.info("翻译处理器已初始化")
                except Exception as e:
                    logger.warning(f"初始化翻译处理器失败: {e}")
            
        except ImportError as e:
            logger.error(f"导入 funasr-onnx 失败: {e}")
            logger.error("请安装必要的依赖: pip install -U funasr-onnx")
            raise
        except Exception as e:
            logger.error(f"初始化本地 SenseVoiceSmall 模型失败: {e} {traceback.format_exc()}")
            raise
    
    def _save_audio_to_temp_file(self, audio_buffer):
        """将音频缓冲区保存为临时文件"""
        try:
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file_path = temp_file.name
            temp_file.close()
            
            # 将音频数据写入临时文件
            audio_buffer.seek(0)  # 确保从头开始读取
            audio_data, sample_rate = sf.read(audio_buffer)
            sf.write(temp_file_path, audio_data, sample_rate)
            
            return temp_file_path
        except Exception as e:
            logger.error(f"保存音频到临时文件失败: {e}")
            return None
    
    def process_audio(self, audio_buffer, mode="transcriptions", prompt=""):
        """处理音频（转录或翻译）
        
        Args:
            audio_buffer: 音频数据缓冲
            mode: 'transcriptions' 或 'translations'，决定是转录还是翻译
            prompt: 提示词（本地模式下不使用）
        
        Returns:
            tuple: (结果文本, 错误信息)
            - 如果成功，错误信息为 None
            - 如果失败，结果文本为 None
        """
        temp_file_path = None
        try:
            start_time = time.time()
            
            # 保存音频到临时文件
            temp_file_path = self._save_audio_to_temp_file(audio_buffer)
            if not temp_file_path:
                return None, "保存音频到临时文件失败"
            
            logger.info(f"正在使用本地 SenseVoiceSmall 模型处理音频... (模式: {mode})")
            
            # 使用模型进行推理
            wav_files = [temp_file_path]
            language = "auto"  # 自动检测语言
            
            # 执行推理
            results = self.model(wav_files, language=language, use_itn=True)
            
            # 处理结果
            if results and len(results) > 0:
                # 应用后处理
                processed_results = [self.rich_transcription_postprocess(r) for r in results]
                result = processed_results[0] if processed_results else ""
                
                # 如果是翻译模式且翻译处理器可用
                if mode == "translations" and self.translate_processor:
                    result = self.translate_processor.translate(result)
                
                logger.info(f"本地模型处理成功 ({mode}), 耗时: {time.time() - start_time:.1f}秒")
                logger.info(f"识别结果: {result}")
                
                # 清理临时文件
                if temp_file_path:
                    try:
                        os.unlink(temp_file_path)
                    except Exception as e:
                        logger.warning(f"删除临时文件失败: {e}")
                
                return result, None
            else:
                error_msg = "模型未返回有效结果"
                logger.error(error_msg)
                return None, error_msg
                
        except Exception as e:
            error_msg = f"❌ {str(e)}"
            logger.error(f"音频处理错误: {str(e)}", exc_info=True)
            
            # 尝试清理临时文件
            if temp_file_path:
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass
                
            return None, error_msg
        finally:
            # 确保关闭音频缓冲区
            if audio_buffer and hasattr(audio_buffer, 'close'):
                audio_buffer.close() 
