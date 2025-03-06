import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
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
export class ChatComponent implements OnInit {
  @ViewChild('fileInput') fileInput: ElementRef;
  topic: string = '';
  chat_response: string;
  chat_answer: string;
  question: string = '';
  commands: string[] = [
    'Provide a summary of the research report for {company}.',
    'What are the analyst ratings for {company}?',
    'What are the risks for {company}?',
    'What are the opportunities for {company}?',
    'What is the target price for {company}?',
    'How does the analyst justify the target price for {company}?',
    'Who are the competitors of {company}?',
    'Provide the key financial metrics used by the analyst to rate {company} and for each of these metrics provide the values',
    'Provide the details of the analyst who wrote the report for {company}, their affiliation, and the date of the report.',
  ];
  companies: string[] = [];//['3P Learning', 'ABB', 'Apple Inc', 'CBS Corporation', 'Duke Energy', 'Imperial Oil Limited',
    //'Premier Foods', 'Sanofi', 'Schneider Electric', 'The Walt Disney Company', 'Virgin Money Holdings'];
  selectedCompany: string;
  updatedCommands: string[] = [];
  uploadedCompanyName: string='';
  uploadError: string = '';
  selectedFile: File | null = null;

  constructor(private route: ActivatedRoute, private chatService: ChatService, private http: HttpClient, private router: Router) {
    //this.loadChatHistory();
  }
  ngOnInit(): void {
    this.getCompaniesList();
    if (this.companies.length > 0) {
      this.selectedCompany = this.companies[0];
    }
    this.updateCommands();
  }

  //invoke a getCompaniesList method to get the list of companies
  getCompaniesList() {
    console.log('Invoking getCompaniesList');
    this.chatService.getCompaniesList().subscribe((response: any) => {
      console.log('response:', JSON.stringify(response));
      //response.companies is an array of company names that looks like this:
      //response:"[\"3P Learning\"]". We need to parse it to get the array of company names
      //for each company in the array, add it to the companies array using addCompany method
      response.forEach((company: string) => {
        this.addCompany(company);
      }
      );
      console.log('companies:', this.companies);
      //sort the companies in alphabetical order
      this.companies.sort();


      // Automatically select the first company in the list
      if (this.companies.length > 0) {
        this.selectedCompany = this.companies[0];
        this.updateCommands();
      }
    },
    (error: any) => {
      console.error('Error:', error);
    });
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

  addCompany(company: string) {
    this.companies.push(company);
    this.sortCompaniesList();
  }

  updateCompany(index: number, newCompany: string) {
    if (index >= 0 && index < this.companies.length) {
      this.companies[index] = newCompany;
    }
  }

  removeCompany(index: number) {
    if (index >= 0 && index < this.companies.length) {
      this.companies.splice(index, 1);
    }
  }

  selectCompany(company: string) {
    //print the seleted company
    console.log('Selected company: ' + company);
    this.selectedCompany = company;
    this.updateCommands();
  }

  //Add a method to sort the companies list
  sortCompaniesList() {
    this.companies.sort();
  }

  insertCommand(command: string) {
    this.question = command;
  }

  updateCommands() {
    this.updatedCommands = this.commands.map(command => command.replace('{company}', this.selectedCompany || ''));
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

  clearUploadError(): void {
    this.uploadError = '';
  }

  uploadPdf(): void {
    this.clearUploadError(); // Clear any previous error messages

    const fileInput = this.fileInput.nativeElement;
    const file: File = fileInput.files[0];

    if (!this.uploadedCompanyName) {
      this.uploadError = 'Error - Please enter a company name.';
      return;
    }
    if (!file) {
      this.uploadError = 'Please select a PDF file.';
      return;
    }
    this.uploadError = ''; // Clear any previous error messages

    console.log('Selected file:', file.name);
    console.log('Associated company:', this.uploadedCompanyName);
    this.selectedFile = file; // Store the selected file

    console.log('Uploading file:', this.selectedFile.name);
    console.log('Associated company:', this.uploadedCompanyName);
    // Add uploaded company name to the list of companies
    this.addCompany(this.uploadedCompanyName);
    // Send the file to the server
    this.chatService.uploadFile(this.selectedFile).subscribe(
      (response: any) => {
        console.log('response: ' + response.answer);
        this.chat_answer = response.answer;
        this.chatService.addMessage({
          sender: 'System',
          content: `The file "${this.selectedFile?.name}" for company "${this.uploadedCompanyName}" was uploaded successfully.`,
          timestamp: new Date().toLocaleString(),
          sequence: this.messages.length + 1,
        });
        this.scrollToBottom();

        // Reset the file input value to allow re-selection of the same file
        this.fileInput.nativeElement.value = '';
        this.selectedFile = null; // Clear the selected file
      },
      (error: any) => {
        console.error('Error:', error);
      }
    );
  }
}