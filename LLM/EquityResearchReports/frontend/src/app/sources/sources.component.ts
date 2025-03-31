import { Component, Input, Output, OnInit, EventEmitter, ViewChild, ElementRef } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { SourcesService } from './sources.service';
import { Router } from '@angular/router';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-sources',
  templateUrl: './sources.component.html',
  styleUrls: ['./sources.component.scss']
})
export class SourcesComponent implements OnInit {
  @ViewChild('fileInput') fileInput: ElementRef;
  @Input() selectedCompany: string = ''; // Input from ChatComponent
  @Output() addCompanyEvent = new EventEmitter<string>();
  //Emit the list of companies to the chat component
  @Output() getCompaniesEvent = new EventEmitter<string[]>();
  @Output() deleteCompanyEvent = new EventEmitter<string>();; // EventEmitter to notify ChatComponent
  sources: any[] = []; // List of sources
  newSource: any = { company_name: '', pdf: '', status: 'active' }; // New source object
  editedSource: any = null; // Source being edited
  uploadStatus: string = ''; // Status message for file upload
  private companies: string[] = [];// Dynamically populated from the backend

  constructor(private route: ActivatedRoute, private sourcesService: SourcesService, private http: HttpClient, private router: Router) {
      //this.loadChatHistory();
  }

  ngOnInit(): void {
    this.loadSources('all'); // Load sources when the component initializes
  }

  // Load sources from the backend
  loadSources(status: string) {
    this.sourcesService.getSources().subscribe(
        (data: any) => {
          this.sources = data;
          // Sort the sources in alphabetical order of company name.
          this.sources.sort((a, b) => a.company_name.localeCompare(b.company_name));
          console.log('Sources:', this.sources);
          // Filter out the sources based on the provided status if the status is not 'all'
          if (status !== 'all') {
            this.sources = this.sources.filter(source => source.status === status);
          }
          // Extract the company names from the sources
          let companies = this.sources.map((source) => source.company_name);
          // Emit the event to get the companies
            this.getCompaniesEvent.emit(companies.filter(company => 
              this.sources.some(source => source.company_name === company && source.status === 'active')
            ));
        },
        (error: any) => {
          console.error('Error loading sources:', error);
        }
      );
  }

   // Method to emit the getCompaniesEvent
   getCompanies(): void {
    this.getCompaniesEvent.emit(this.companies);
  }

  clearUploadError(): void {
    this.uploadStatus = '';
  }

   // Emit the event to add a new company
   addCompany(): void {
    if (!this.newSource.company_name) {
      console.error('Company name is required.');
      return;
    }
    console.log(`Emitting event to add company: ${this.newSource.company_name}`);
    this.addCompanyEvent.emit(this.newSource.company_name); // Emit the company name as a string
  }
      
  uploadPdf(): void {
    this.clearUploadError(); // Clear any previous error messages

    const fileInput = this.fileInput.nativeElement;
    const file: File = fileInput.files[0];

    if (!this.newSource.company_name) {
      this.uploadStatus = 'Error - Please enter a company name.';
      return;
    }
    if (!this.newSource.pdf) {
      this.uploadStatus = 'Please select a PDF file.';
      return;
    }
    //ensure that file name is same as uploaded file name
    if (this.newSource.pdf !== file.name) {
      this.uploadStatus = 'Error - Please select the correct PDF file.';
      return;
    }
    //ensure that the filename entered is a pdf file
    if (!file.name.endsWith('.pdf')) {
      this.uploadStatus = 'Error - Please select a PDF file.';
      return;
    }
    
    this.uploadStatus = ''; // Clear any previous error messages

    console.log('Selected file:', file.name);
    console.log('Associated company:', this.newSource.company_name);
    console.log('Uploading file:', this.newSource.pdf);
    // Add uploaded company name to the list of companies
    this.addCompany();
    // Send the file to the server
    this.sourcesService.uploadFile(file,this.newSource.company_name).subscribe(
      (response: any) => {
        console.log('response: ' + response.answer);
        this.uploadStatus = `File "${file.name}" uploaded successfully.`;
        this.loadSources('all'); // Reload the sources list
        // Reset the file input value to allow re-selection of the same file
        this.fileInput.nativeElement.value = '';
        this.newSource.pdf = null; // Clear the selected file
        //clear the uploaded company name
        this.newSource.company_name = '';
      },
      (error: any) => {
        this.uploadStatus = 'Error - Failed to upload the file. Please try again.';
        console.error('Error:', error);
      }
    );
  }

  // // Add a new source
  // addSource() {
  //   // Replace with actual API call
  //   fetch('http://localhost:5003/source', {
  //     method: 'POST',
  //     headers: { 'Content-Type': 'application/json' },
  //     body: JSON.stringify(this.newSource),
  //   })
  //     .then((response) => response.json())
  //     .then((data) => {
  //       this.sources.push(data); // Add the new source to the list
  //       this.newSource = { company_name: '', pdf: '', status: 'active' }; // Reset the form
  //     })
  //     .catch((error) => console.error('Error adding source:', error));
  // }

  // Edit an existing source
  editSource(source: any) {
    this.editedSource = { ...source }; // Clone the source to edit
  }

  // Save the edited source
  saveSource() {
    // Replace with actual API call
    fetch(`http://localhost:5003/source/${this.editedSource.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(this.editedSource),
    })
      .then((response) => response.json())
      .then((data) => {
        // Update the source in the list
        const index = this.sources.findIndex((s) => s.id === data.id);
        if (index !== -1) {
          this.sources[index] = data;
        }
        this.editedSource = null; // Clear the edit form
      })
      .catch((error) => console.error('Error saving source:', error));
  }

  // Delete a source
  deleteSource(sourceId: number, companyName: string) {
    // Replace with actual API call
    fetch(`http://localhost:5003/source/${sourceId}`, { method: 'DELETE' })
    .then((response) => {
      if (response.ok) {
        console.log(`Source with ID ${sourceId} deleted successfully.`);
        this.loadSources('all'); // Refresh the list of sources
        console.log(`Emitting event to delete company: ${companyName}`);
        this.deleteCompanyEvent.emit(companyName); // Emit the company name as a string
    
      } else {
        console.error(`Failed to delete source with ID ${sourceId}.`);
      }
    })
    .catch((error) => {
      console.error('Error deleting source:', error);
    });
  }
}