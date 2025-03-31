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

    private sourcesURL: string;
    private baseURL: string;

    constructor(private httpClient: HttpClient, private configService: RuntimeConfigService) {
      this.sourcesURL = configService.sourcesUrl;
      this.baseURL = configService.baseUrl;
      //if sourcesURL is null or configService is not initialized, use environment variable
      if (!this.sourcesURL) {
        this.sourcesURL = environment.sourcesURL;
      }
      console.log('sourcesURL is set to ' + this.sourcesURL);
    }

    uploadFile(file: File, uploadedCompanyName: String): Observable<any> {
        const formData = new FormData();
        formData.append('file', file, file.name);
        // Add uploadedCompanyName as a query parameter
        const params = new HttpParams().set('company_name', uploadedCompanyName.toString());

        //TODO: Need to change this to use the sourcesURL but this will also require moving the backend functionality from chat seqrvice to sources service.
        return this.httpClient.post(`${this.baseURL}/upload`, formData, { params });
    }

    getActiveSources(): Observable<any> {
      return this.httpClient.get(`${this.sourcesURL}/source/active`).pipe(
          catchError((error: any) => {
              console.error('Error fetching sources:', error);
              return throwError(error);
            })
        );
    }

    getSources(): Observable<any> {
        return this.httpClient.get(`${this.sourcesURL}/source`).pipe(
            catchError((error: any) => {
                console.error('Error fetching sources:', error);
                return throwError(error);
            })
        );
    }
}