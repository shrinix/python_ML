package net.siyengar.model;

// import lombok.Data;
// import lombok.experimental.Accessors;

//@Data
//@Accessors(chain = true)
public class ChatResponse {

  private String message;

  public String getMessage() {
    return message;
  }

  public void setMessage(String message) {
    this.message = message;
  }

  //Add toString method
  @Override
  public String toString() {
    return "ChatResponse [message=" + message + "]";
  }
}


