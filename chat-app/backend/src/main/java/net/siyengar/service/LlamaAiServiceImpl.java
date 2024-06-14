package net.siyengar.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import net.siyengar.model.ChatResponse;

import org.springframework.ai.chat.ChatClient;


@Service
public class LlamaAiServiceImpl implements LlamaAiService {

  private final ChatClient chatClient;

  @Autowired
  public LlamaAiServiceImpl(ChatClient chatClient) {
    this.chatClient = chatClient;
  }

  @Override
  public ChatResponse generateMessage(String promptMessage) {
    final String llamaMessage = chatClient.call(promptMessage);
    ChatResponse response = new ChatResponse();
    response.setMessage(llamaMessage);
    return response;
  }

  @Override
  public ChatResponse generateJoke(String topic) {
    final String llamaMessage = chatClient.call(String.format("Tell me a joke about %s", topic));
    ChatResponse response = new ChatResponse();
    response.setMessage(llamaMessage);
    return response;
  }
}
