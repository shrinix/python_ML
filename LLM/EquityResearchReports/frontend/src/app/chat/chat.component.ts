import { Component, OnInit, ViewChild, ElementRef,AfterViewInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { ChatService } from './chat.service';
import { Router } from '@angular/router';
import { ActivatedRoute } from '@angular/router';
import { SourcesComponent } from '../sources/sources.component';
import { SourcesService } from '../sources/sources.service';

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
export class ChatComponent implements OnInit, AfterViewInit {
  // @ViewChild('fileInput') fileInput: ElementRef;
  topic: string = '';
  chat_response: string;
  chat_answer: string;
  chat_source_docs: string;
  question: string = '';
  loading: boolean = false; // Tracks whether the API call is in progress
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
  @ViewChild(SourcesComponent, { static: false }) sourcesComponent?: SourcesComponent; // Reference to the SourcesComponent
  updatedCommands: string[] = [];
  uploadedCompanyName: string = '';
  uploadError: string = '';
  selectedFile: File | null = null;
  ragas_score: string = '';
  ragas_faithfullness: string = '';
  ragas_relevancy: string = '';
  tabNames = {
    chat: 'Chat',
    analytics: 'Analytics',
    sources: 'Sources',
  };
  selectedTab: 'chat' | 'analytics' | 'sources' = 'chat'; // Restrict to specific values // Default tab is 'chat'
  iaReport: { [key: string]: string } = {}; // Property to store the parsed IA report
  iaReportText: string = ''; // Property to store the IA report text

  constructor(private route: ActivatedRoute, private chatService: ChatService, 
    private http: HttpClient, private router: Router,private sourcesService: SourcesService) {
    //this.loadChatHistory();
  }
  ngOnInit(): void {
    console.log('Initializing ChatComponent...');
    // Load companies when the component is initialized
    this.loadCompanies();
    // Set the default tab
    this.setTab('chat'); // Explicitly set the default tab to 'chat'
    this.updateCommands();
  }

  loadCompanies(): void {
    this.sourcesService.getActiveSources().subscribe(
      (data: any) => {
        const sources = data;
        sources.sort((a: any, b: any) => a.company_name.localeCompare(b.company_name));
        this.companies = sources.map((source: any) => source.company_name);
        if (this.companies.length > 0) {
          this.selectedCompany = this.companies[0];
          this.updateCommands();
        }
      },
      (error: any) => {
        console.error('Error loading companies:', error);
      }
    );
  }

  ngAfterViewInit(): void {
    // Trigger the getCompaniesEvent programmatically after the view is initialized
    if (this.sourcesComponent) {
      if (this.sourcesComponent) {
        this.sourcesComponent.getCompanies();
      } else {
        console.warn('SourcesComponent is not initialized.');
      }
    } else {
      console.warn('SourcesComponent is not initialized.');
    }
  }

  setTab(tab: keyof typeof this.tabNames) {
    if (this.tabNames[tab]) {
      this.selectedTab = tab;
      console.log('Selected Tab:', this.tabNames[tab]);
    } else {
      console.warn(`Invalid tab: ${tab}`);
    }
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
    console.log('Setting loading spinner to true');
    this.loading = true; //show loading spinner
    this.chatService.getAnswer(this.question).subscribe((response: any) => {
      console.log('Response received, setting loading to false');
      this.loading = false; //hide spinner
      console.log('response: ' + response.answer);
      if (response.metrics) {
        console.log('metrics: ' + response.metrics);
        //response.metrics is an array that looks like this:
        //{RAGAS Score: 0.97, answer_relevancy: 0.95, faithfulness: 1,
        // response: "The target price for 3P Learning is set at $2.70 per share based on the Discounted Cash Flow (DCF) methodology.",
        // retrieved_contexts: ["14 January 2015 ↵Emerging Companies ↵3P Learning ↵…ur $2.70 DCF. Key inputs: beta 1.2; WACC 10.8%; &", "14 January 2015 ↵Emerging Companies ↵3P Learning ↵…ur $2.70 DCF. Key inputs: beta 1.2; WACC 10.8%; &", "14 January 2015 ↵Emerging Companies ↵3P Learning ↵…ur $2.70 DCF. Key inputs: beta 1.2; WACC 10.8%; &", "14 January 2015 ↵Emerging Companies ↵3P Learning ↵…rix to WACC & TGR.   ↵ ↵PT set equal to $2.70 DCF", "14 January 2015 ↵Emerging Companies ↵3P Learning ↵…rix to WACC & TGR.   ↵ ↵PT set equal to $2.70 DCF", "14 January 2015 ↵Emerging Companies ↵3P Learning ↵…rix to WACC & TGR.   ↵ ↵PT set equal to $2.70 DCF"],
        // user_input: "What is the target price for 3P Learning?"}
        //we can access the RAGAS score, answer_relevancy, and faithfulness from the response
        //set the ragas_faithfullness and ragas_relevancy properties to the corresponding values in the response      this
        this.ragas_score = response.metrics[0]['RAGAS Score'];
        this.ragas_faithfullness = response.metrics[0].faithfulness;
        this.ragas_relevancy = response.metrics[0].answer_relevancy;
      }

      this.chat_answer = response.answer;
      this.chat_source_docs = response.source_documents;

      // Format the bot response into a multiline format if it contains a numbered list
      const items = this.chat_answer.split(/(\d+\.\s)/g).filter(Boolean); // Split by numbered items
      const formattedItems: string[] = [];
      for (let i = 0; i < items.length; i++) {
        if (/\d+\.\s/.test(items[i])) {
          // If the current item is a number (e.g., "2. "), combine it with the next item
          const numberedLine = items[i] + (items[i + 1] || '');
          formattedItems.push(numberedLine.trim());
          i++; // Skip the next item since it's already combined
        } else {
          // If it's not a number, just add it
          formattedItems.push(items[i].trim());
        }
      }

      // Join the formatted items with <br> for HTML line breaks
      this.chat_answer = formattedItems.join('<br>');

      // Ensure currency values are not split
      this.chat_answer = this.chat_answer.replace(/(\b[A-Z]{3})<br>(\d+)/g, '$1 $2');

      const timestamp = new Date().toLocaleString(undefined, { timeZoneName: 'short' });
      const sequence = this.messages.length + 1;
      const userMessage: ChatMessage = { sender: 'User', content: this.question, timestamp, sequence }
      const botMessage: ChatMessage = { sender: 'Bot', content: this.chat_answer, timestamp, sequence: sequence + 1, isHtml: true, cssClass: 'bot-response' }

      // this.question = ''; // Clear the input box
      // this.scrollToBottom();

      this.chatService.addMessage(userMessage);

      //Add the source documents to the chat
      if (this.chat_source_docs) {
        botMessage.content += '<br>Sources:--><br>' + JSON.stringify(this.chat_source_docs);
      }
      this.chatService.addMessage(botMessage);
      this.scrollToBottom()
    },
      (error: any) => {
        console.error('Error:', error);
        this.loading = false; // Set loading to false even if the API call fails
        this.chatService.addMessage({
          sender: 'System',
          content: 'An error occurred while processing your request. Please try again later.',
          timestamp: new Date().toLocaleString(),
          sequence: this.messages.length + 1,
        });
      });
  }

  // Handle the event from SourcesComponent
  handleAddCompany(companyName: string): void {
    if (!this.companies.includes(companyName)) {
      this.addCompany(companyName);
      console.log(`Company added: ${companyName}`);
    } else {
      console.log(`Company already exists: ${companyName}`);
    }
  }

  // Handle the event from SourcesComponent
  handleDeleteCompany(companyName: string): void {
    if (!this.companies.includes(companyName)) {
      this.removeCompanyByName(companyName);
      console.log(`Company added: ${companyName}`);
    } else {
      console.log(`Company already exists: ${companyName}`);
    }
  }

  //Handle the companies list event from SourcesComponent
  handleGetCompanies(companies: string[]): void {
    console.log('Companies received:', companies);
    this.companies = companies;
    this.sortCompaniesList();
    console.log('Sorted companies:', this.companies);
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

  removeCompanyByName(name: string) {
    const index = this.companies.indexOf(name);
    if (index >= 0) {
      this.removeCompany(index);
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
    this.iaReport = {}; // Clear the IA report
    this.iaReportText = ''; // Clear any existing IA report text
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

  // clearUploadError(): void {
  //   this.uploadError = '';
  // }

  hasKeys(obj: any): boolean {
    return obj && Object.keys(obj).length > 0;
  }

  // uploadPdf(): void {
  //   this.clearUploadError(); // Clear any previous error messages

  //   const fileInput = this.fileInput.nativeElement;
  //   const file: File = fileInput.files[0];

  //   if (!this.uploadedCompanyName) {
  //     this.uploadError = 'Error - Please enter a company name.';
  //     return;
  //   }
  //   if (!file) {
  //     this.uploadError = 'Please select a PDF file.';
  //     return;
  //   }
  //   this.uploadError = ''; // Clear any previous error messages

  //   console.log('Selected file:', file.name);
  //   console.log('Associated company:', this.uploadedCompanyName);
  //   this.selectedFile = file; // Store the selected file

  //   console.log('Uploading file:', this.selectedFile.name);
  //   console.log('Associated company:', this.uploadedCompanyName);
  //   // Add uploaded company name to the list of companies
  //   this.addCompany(this.uploadedCompanyName);
  //   // Send the file to the server
  //   this.chatService.uploadFile(this.selectedFile,this.uploadedCompanyName).subscribe(
  //     (response: any) => {
  //       console.log('response: ' + response.answer);
  //       this.chat_answer = response.answer;
  //       this.chatService.addMessage({
  //         sender: 'System',
  //         content: `The file "${this.selectedFile?.name}" for company "${this.uploadedCompanyName}" was uploaded successfully.`,
  //         timestamp: new Date().toLocaleString(),
  //         sequence: this.messages.length + 1,
  //       });
  //       this.scrollToBottom();

  //       // Reset the file input value to allow re-selection of the same file
  //       this.fileInput.nativeElement.value = '';
  //       this.selectedFile = null; // Clear the selected file
  //       //clear the uploaded company name
  //       this.uploadedCompanyName = '';
  //     },
  //     (error: any) => {
  //       console.error('Error:', error);
  //       this.chatService.addMessage({
  //         sender: 'System',
  //         content: `Failed to upload the file "${this.selectedFile?.name}". Please try again.`,
  //         timestamp: new Date().toLocaleString(),
  //         sequence: this.messages.length + 1,
  //       });
  //     }
  //   );
  // }

  formatReportContent(content: string): string {
    console.log('Before Formatting content:', content);
  
    // Convert subsection leaders like "**Valuation:**" or "**Valuation**:" to bold text
    content = content.replace(/(?:^|\n)\*\*(.+?)\*\*:?/g, '<strong>$1:</strong>');
  
    // Ensure leaders are followed by their content on the same line
    content = content.replace(/<\/strong>\s*\n\s*(?!-|\d+\.)/g, '</strong> '); // Remove line breaks between leaders and their content unless it's a list
  
    // Group leaders and their associated content into a single paragraph
    content = content.replace(/(<strong>.+?<\/strong>)([\s\S]*?)(?=(<strong>|$))/g, (match, leader, body) => {
      // Trim and clean up the body content
      const formattedBody = body
        .trim()
        .replace(/\n+/g, ' ') // Remove extra line breaks within the body
        .replace(/-\s+/g, '- '); // Ensure proper spacing for bullet points
      return `${leader} <p>${formattedBody}</p>`;
    });
  
    // Handle numbered lists (e.g., "1. item", "2. item")
    content = content.replace(/(?:^|\n)(\d+)\. (.+)/g, '<li>$2</li>'); // Convert "1. item" to list items
    content = content.replace(/(<li>(?:.|\n)*?<\/li>)/g, '<ol>$1</ol>'); // Wrap all list items in a single <ol>
  
    //trim lines with whitespaces in the beginning and end
    content = content.replace(/^\s+|\s+$/g, '');

    // Handle bulleted lists (e.g., "- item")
    content = content.replace(/(?:^|\n)- (.+)/g, '<li>$1</li>'); // Convert "- item" to list items
    content = content.replace(/(<li>(?:.|\n)*?<\/li>)/g, '<ul>$1</ul>'); // Wrap all list items in a single <ul>
  
    // Remove misplaced list annotations around leaders
    content = content.replace(/<li>(<strong>.+?<\/strong>)<\/li>/g, '$1'); // Remove <li> tags around leaders
  
    // Remove any extra blank lines caused by formatting
    // Replace multiple blank lines with no blank line
    content = content.replace(/\n{2,}/g, '\n');
    
    console.log('Formatted content:', content);
    return content;
  }

  // Method to trigger the backend API
  generateIAReport(company: string) {
    if (!company) {
      this.iaReportText = 'Please select a company before generating the IA Report.';
      return;
    }

    //clear the existing IA report
    this.iaReport = {};
    this.iaReportText = ''; // Clear the IA report text

    this.chatService.generateIAReport(company).subscribe(
      (response) => {
        console.log('IA Report generated successfully:', response);
        // Parse the JSON string response
        // Extract the IA_report object from the response
        if (response && response.IA_report) {
         // Preprocess each section's content
        for (const key in response.IA_report) {
          if (response.IA_report.hasOwnProperty(key)) {
            response.IA_report[key] = this.formatReportContent(response.IA_report[key]);
          }
        }
          this.iaReport = response.IA_report; // Store the formatted IA report
        } else {
          this.iaReportText = 'Invalid response format.';
        }
      },
      (error) => {
        console.error('Error generating IA Report:', error);
        alert('Failed to generate IA Report.');
      }
    );
  }
}