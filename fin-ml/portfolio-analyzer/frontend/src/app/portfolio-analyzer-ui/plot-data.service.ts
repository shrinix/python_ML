import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { HttpClient, HttpHeaders } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class PlotDataService {
  private portfolioDataAPIURL = 'http://127.0.0.1:5000/api/portfolio-plot-data'; 
  private tickerMergedDataAPIURL = 'http://127.0.0.1:5000/api/ticker-plot-data'; 

  constructor(private httpclient: HttpClient) { }

  getTimeSeriesPlotData(): Observable<any> {
    const headers = new HttpHeaders({
      'Content-Type': 'application/json'
    });
    return this.httpclient.get<any>(this.tickerMergedDataAPIURL, { headers });
  }

  getPortfolioAnalysisData(): Observable<any> {
    const headers = new HttpHeaders({
      'Content-Type': 'application/json'
    });
    return this.httpclient.get<any>(this.portfolioDataAPIURL, { headers });
  }
}