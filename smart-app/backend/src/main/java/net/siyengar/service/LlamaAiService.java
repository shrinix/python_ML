package net.siyengar.service;
import net.siyengar.model.ChatResponse;

public interface LlamaAiService {

    ChatResponse generateMessage(String prompt);
    ChatResponse generateJoke(String topic);
  }