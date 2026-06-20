from load_config import load_configuration
from random import randrange
class Colors:
    USER = '\033[94m'
    PARTIAL = '\033[93m'
    ASSISTANT = '\033[92m'
    ERROR = '\033[91m'
    RESET = '\033[0m'

loaded = load_configuration()
cfg = loaded["cfg"]

print(f"\n{Colors.ASSISTANT}🔊 {cfg.get("llm_speaking", "Llm is speaking...")}{Colors.RESET}")
print(f"{Colors.ERROR}[ERROR] {cfg.get("micr_err", "Unable to unmute microphone (stream inactive).")}")
print(f"{Colors.USER}🎤 {cfg.get("user_speaking", "It's your turn to speak")}{Colors.RESET}\n")           

print(f"{Colors.ERROR}❌ {cfg.get("mic_not_avail", "Error: Microphone not turned on or unavailable.")}")
print(f"[ERROR] {cfg.get("err_details", "Details")}: ")
print(f"[ERROR] {cfg.get("db_init_fail", "Database initialization failed")}: ")
print(f"{Colors.ERROR}[ERROR] {cfg.get("ins_msg_err", "Error writing to messages table.")}: ")
print(f"[DB] {cfg.get("conn_closed", "Connection closed automatically.")}")
print(f"[DEBUG][embed_text] {cfg.get("embed_time", "Embedding time:")} sec ({cfg.get("txt_length", "text length")} {cfg.get("payload_chars", "chars")})")
print(f"{Colors.ERROR}{cfg.get("embed_fail", "Embedding failed:")}")
print(f"{Colors.ERROR}[Summarizer] {cfg.get("llm_call_failed", "Ollama call failed:")} ")
print(f"[Summarizer] {cfg.get("summ_saved", "Stored summary:")} ({cfg.get("turns_compr", "turns compressed")}).")
identity_message = f"{cfg.get("identity_llm", "Important note: I am Zira")}{cfg.get("identity_user", "You are Marco.")}"
print(f"{Colors.ERROR}[ERROR] {cfg.get("payload_err", "Payload error")} ")
print(f"{Colors.ASSISTANT}Assistant: {cfg.get("keyb_type_msg", "Okay, you can type from the keyboard.")}{Colors.RESET}")
print(f"{Colors.ASSISTANT}Assistant: {cfg.get("activ_welc_msg", "System active. Waiting for activation word.")}{Colors.RESET}")
print(f"{Colors.PARTIAL}{cfg.get("partial_understood", "Understood (partially):")} {Colors.RESET}")

misc_array = cfg.get("acknowledgements")
print (misc_array[randrange(0, len(misc_array))])
print(cfg.get("llm_stopped", "Assistant terminated."))

cfg = loaded.get("cfg")
print (cfg)