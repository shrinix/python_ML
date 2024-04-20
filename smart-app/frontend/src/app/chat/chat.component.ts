import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormsModule} from '@angular/forms';
import { ChatService } from './chat.service';
import { Router } from '@angular/router';
import { ActivatedRoute } from '@angular/router';


interface ChatMessage {
  sender: string;
  content: string;
}

@Component({
  selector: 'your-app-root',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss'],
})
export class ChatComponent {
  messages: ChatMessage[] = [];
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
    console.log('Invoking generateAnswer from ChatComponent: '+question);
    this.chatService.getAnswer(this.question).subscribe((response:any) => {
      console.log ('response: '+response.message); 
      this.chat_answer = response.message;
    },
    (error:any) => {
      console.error('Error:', error);
    });
  }

  generateJoke(topic: string){
    console.log('Invoking generateJoke from ChatComponent: '+topic);
    this.chatService.getJokeByTopic(this.topic).subscribe( data => {
      //get the joke from the response
      if (data) {
        console.log('Joke: '+data);
        this.chat_response = data.message;
      } else {
        console.log('Response was empty');
      }
      },
      error => {
        console.error('Error:', error);
      }
    );

  }

  // sendMessage() {

  //   if (!this.newMessage.trim()) return;
  //   console.log(this.newMessage.trim());

  //   const headers = { 'Content-Type': 'application/json' }; // Add headers

  //   this.http.post<ChatMessage>('http://localhost:8080/generate/joke/' + this.newMessage.trim(), { content: this.newMessage }, { headers }).subscribe({ // Pass headers in the request
  //     next: (data) => {
  //     this.messages.push(data);
  //     this.newMessage = '';
  //     },
  //     error: (error) => {
  //     console.error('Error sending message:', error);
  //     }
  //   });
  // }
}