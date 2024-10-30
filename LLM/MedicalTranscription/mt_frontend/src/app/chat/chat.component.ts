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
  isHtml?: boolean; // Optional property to indicate if the content is HTML
  cssClass?: string; // Optional property to specify a CSS class for the message
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
  commands: string[] = [

    '1. Provide a summary of the health status of {Patient}.',
    '2. What is the past medical history of {Patient}?',
    '3. What is the past surgical history of {Patient}?',
    '4. What is the social history of {Patient}?',
    '5. What is the family history of {Patient}?',
    '6. What are the current medications of {Patient}?',
    '7. What are the allergies of {Patient}?',
    '8. What is the eating history of {Patient}?',
    '9. What is the review of systems for {Patient}?',
    '10. What are the physical examination details for {Patient}?',
  ];
  patients: string[] = ['Patient001', 'PatientABC'];
  selectedPatient: string;
  updatedCommands: string[] = [];

  constructor(private route: ActivatedRoute, private chatService: ChatService, private http: HttpClient, private router: Router) {
    //this.loadChatHistory();
  }
  ngOnInit(): void {
    if (this.patients.length > 0) {
      this.selectedPatient = this.patients[0];
    }
    this.updateCommands();
  }

  generateAnswer(question: string) {
    console.log('Invoking generateAnswer from ChatComponent: ' + question);
    this.chatService.getAnswer(this.question).subscribe((response: any) => {
      console.log('response: ' + response.answer);
      this.chat_answer = response.answer;
      
      // Format the bot response into a multiline format if it contains a numbered list
      const items = this.chat_answer.split(/(\d+\.\s+)/g).filter(Boolean);
      for (let i = 1; i < items.length; i += 2) {
        items[i] = items[i] + (items[i + 1] || '');
        if (items[i + 1]) {
          items.splice(i + 1, 1);
        }
      }
      
      // Ensure currency values are not split
      this.chat_answer = items.join('<br>').replace(/(\b[A-Z]{3})<br>(\d+)/g, '$1 $2'); // Use <br> for HTML line breaks

      const timestamp = new Date().toLocaleString(undefined, {timeZoneName: 'short'});
      const sequence = this.messages.length + 1;
      const userMessage: ChatMessage = {sender: 'User', content: this.question, timestamp, sequence }
      const botMessage: ChatMessage ={ sender: 'Bot', content: this.chat_answer, timestamp, sequence: sequence + 1, isHtml: true, cssClass: 'bot-response' }

      // this.question = ''; // Clear the input box
      // this.scrollToBottom();
      
      this.chatService.addMessage(userMessage);
      this.chatService.addMessage(botMessage);
      this.scrollToBottom()
    },
      (error: any) => {
        console.error('Error:', error);
      });
  }

  addPatient(patient: string) {
    this.patients.push(patient);
  }

  updatePatient(index: number, newPatient: string) {
    if (index >= 0 && index < this.patients.length) {
      this.patients[index] = newPatient;
    }
  }

  removePatient(index: number) {
    if (index >= 0 && index < this.patients.length) {
      this.patients.splice(index, 1);
    }
  }

  selectPatient(Patient: string) {
    //print the seleted Patient
    console.log('Selected patient: ' + Patient);
    this.selectedPatient = Patient;
    this.updateCommands();
  }

  insertCommand(command: string) {
    this.question = command;
  }

  updateCommands() {
    this.updatedCommands = this.commands.map(command => command.replace('{Patient}', this.selectedPatient || ''));
  }

  sendCommand(command: string) {
    this.question = command;
    this.generateAnswer(command);
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