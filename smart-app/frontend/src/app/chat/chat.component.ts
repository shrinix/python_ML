import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormsModule} from '@angular/forms';
import { ChatService } from './chat.service';
import { Router } from '@angular/router';
import { ActivatedRoute } from '@angular/router';


export interface ChatMessage {
  sender: string;
  content: string;
  timestamp: string;
  sequence: number;
}

@Component({
  selector: 'your-app-root',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss'],
})
export class ChatComponent {
  topic: string = '';
  chat_response: string;
  chat_answer: string;
  question: string = '';

  constructor(private route: ActivatedRoute, private chatService: ChatService, private http: HttpClient, private router: Router) {
    //this.loadChatHistory();
  }
  ngOnInit(): void {
    // this.topic = this.route.snapshot.params['topic'];
    // this.question = this.route.snapshot.params['question'];
    // this.generateJoke(this.topic);
    // this.generateAnswer(this.question);
  }

  generateAnswer(question: string) {
    console.log('Invoking generateAnswer from ChatComponent: ' + question);
    this.chatService.getAnswer(this.question).subscribe((response: any) => {
      console.log('response: ' + response.message);
      this.chat_answer = response.message;
      
      const timestamp = new Date().toLocaleString(undefined, {timeZoneName: 'short'});
      const sequence = this.messages.length + 1;
      const userMessage: ChatMessage = {sender: 'User', content: this.question, timestamp, sequence }
      const botMessage: ChatMessage ={ sender: 'Bot', content: this.chat_answer, timestamp, sequence: sequence + 1 }

      this.question = ''; // Clear the input box
      this.scrollToBottom();
      
      this.chatService.addMessage(userMessage);
      this.chatService.addMessage(botMessage);
    },
      (error: any) => {
        console.error('Error:', error);
      });
  }

  scrollToBottom(): void {
    const element = document.querySelector('.response-box');
    if (element) {
      element.scrollTop = element.scrollHeight;
    }
  }

  get messages() {
    return this.chatService.messages;
  }
}