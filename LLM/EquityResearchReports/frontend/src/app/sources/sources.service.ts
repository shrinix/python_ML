import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { HttpParams } from '@angular/common/http';
import { map } from 'rxjs/operators';
import { catchError } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { RuntimeConfigService } from '../runtime-config.service';

@Injectable({
  providedIn: 'root'
})
export class SourcesService {

    private baseURL: string;

    constructor(private httpClient: HttpClient, private configService: RuntimeConfigService) {
      this.baseURL = configService.baseUrl;
      //if baseURL is null or configService is not initialized, use environment variable
      if (!this.baseURL) {
        this.baseURL = environment.baseURL;
      }
      console.log('baseURL is set to ' + this.baseURL);
    }

    uploadFile(file: File, uploadedCompanyName: String): Observable<any> {
        const formData = new FormData();
        formData.append('file', file, file.name);
        // Add uploadedCompanyName as a query parameter
        const params = new HttpParams().set('company_name', uploadedCompanyName.toString());

        return this.httpClient.post(`${this.baseURL}/upload`, formData, { params });
    }
}