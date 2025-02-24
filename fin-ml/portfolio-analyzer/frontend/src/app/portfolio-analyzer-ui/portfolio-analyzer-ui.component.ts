import { Component, OnInit, AfterViewInit, ViewChild, importProvidersFrom } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';
import * as Plotly from 'plotly.js-dist';
import { MatTabsModule, MatTabGroup } from '@angular/material/tabs';
import { PlotDataService } from './plot-data.service';

@Component({
  selector: 'app-portfolio-analyzer-ui',
  standalone: true,
  imports: [MatTabsModule, HttpClientModule], // Import MatTabsModule and HttpClientModule here
  templateUrl: './portfolio-analyzer-ui.component.html',
  styleUrls: ['./portfolio-analyzer-ui.component.css'],
  providers: [PlotDataService]
})

export class PortfolioAnalyzerUiComponent implements OnInit , AfterViewInit {
  @ViewChild('tabGroup') tabGroup!: MatTabGroup;

  portfolioData: any;
  mergedStockData: any;
  portfolioWeightsData: any;
  portfolioResultsData: any;
  portfolioWeightsDeltaData: any;

  // Loading flags
  isMergedStockDataLoaded: boolean = false;
  isPortfolioAnalysisDataLoaded: boolean = false;

  constructor(private plotDataService: PlotDataService) { }

  async ngOnInit(): Promise<void>  {
    await this.fetchMergedStockData();
    await this.fetchPortfolioAnalysisData();
  }

  ngAfterViewInit(): void {
    this.tabGroup.selectedTabChange.subscribe(event => {
      this.renderPlots(event.index);
    });
    // Render the plot for the initially active tab
    if (this.tabGroup.selectedIndex !== null) {
      this.renderPlots(this.tabGroup.selectedIndex);
    }
  }

  async fetchMergedStockData(): Promise<void> {
    
    try {
      const data = await this.plotDataService.getTimeSeriesPlotData().toPromise();
      this.mergedStockData = data;
      console.log('Merged stock data:', this.mergedStockData);
      if (this.mergedStockData.length == 0) {
        console.warn('Merged stock data is empty.');
      }
      this.isMergedStockDataLoaded = true;
    } catch (error) {
      console.error('Error parsing merged stock data:', error);
    }
  }

  async fetchPortfolioAnalysisData(): Promise<void> {
      try {
        const data = await this.plotDataService.getPortfolioAnalysisData().toPromise();
        // the data returned is in JSON and is a dictionary of objects
        // data is a dictionary of objects like:
        //   data = {
        //     'portfolio_weights': portfolio_weights_df.to_dict(orient='records'),
        //     'portfolio_value': portfolio_value_df.to_dict(orient='records'),
        //     'portfolio_weights_delta': portfolio_weights_delta.to_dict(orient='records')
        // }
        // parse the data into its components
        this.portfolioWeightsData = data['portfolio_weights'];
        this.portfolioResultsData = data['portfolio_value'];
        this.portfolioWeightsDeltaData = data['portfolio_weights_delta'];
        // log the lenghts of each component
        console.log('Portfolio weights data:', this.portfolioWeightsData);
        console.log('Portfolio results data:', this.portfolioResultsData);
        console.log('Portfolio weights delta data:', this.portfolioWeightsDeltaData);
        if (this.portfolioWeightsData.length == 0) {
          console.warn('Portfolio weights data is empty.');
        }
        if (this.portfolioResultsData.length == 0) {
          console.warn('Portfolio results data is empty.');
        }
        if (this.portfolioWeightsDeltaData.length == 0) {
          console.warn('Portfolio weights delta data is empty.');
        }
        this.isPortfolioAnalysisDataLoaded = true;        
      }
      catch (error) {
        console.error('Error parsing portfolio analysis data:', error);
      }
    }

    renderPlots(tabIndex: number): void {
      if (tabIndex === 0) {
        if (!this.isPortfolioAnalysisDataLoaded) {
          console.warn('Portfolio analysis data is not loaded yet.');
          return;
        }
        this.showPortfolioInfo(this.portfolioResultsData, this.portfolioWeightsDeltaData);
      } 
      else if (tabIndex === 1) {
        if (!this.isMergedStockDataLoaded) {
          console.warn('Merged stock data is not loaded yet.');
          return;
        }
        this.renderTimeSeriesPlot(this.mergedStockData);
        this.renderDailyReturnsPlot(this.mergedStockData);
      } 
      else if (tabIndex === 2) {
        if (!this.isPortfolioAnalysisDataLoaded) {
          console.warn('Portfolio analysis data is not loaded yet.');
          return;
        }
        this.renderPortfolioWeightAnalysisPlot(this.portfolioWeightsData, this.portfolioResultsData);
    }
  }

    tabChanged(event: any): void {
      this.renderPlots(event.index);
    }

    showPortfolioInfo(portfolioResultsData: any, portfolioWeightsDeltaData: any): void {

      const columnsOrder1 = ['Ticker', 'Initial Weight', 'Final Weight'];
      const tableName1 = 'Portfolio Weights Delta';
      const tableDomElementName1 = 'portfolioInfoTable1';
      this.renderTable(tableDomElementName1, tableName1, this.portfolioWeightsDeltaData, columnsOrder1);
      // Access the rendered table element to modify cell values and styles
      const tableElement = document.getElementById(tableDomElementName1);
      if (tableElement) {
        const rows = tableElement.getElementsByTagName('tr');
        for (let i = 1; i < rows.length; i++) { // Start from 1 to skip the header row
            const cells = rows[i].getElementsByTagName('td');
            for (let j = 0; j < cells.length; j++) {
              if (columnsOrder1[j] === 'Initial Weight' || columnsOrder1[j] === 'Final Weight') {
                cells[j].textContent = `${parseFloat(cells[j].textContent || '0').toFixed(2)}%`;
                cells[j].style.textAlign = 'right'; // Right justify the values
              }
            }
        }
      }

      const currency = '$'
      const columnsOrder2 = ['From Date', 'Total Value', 'Adjusted Total Value'];
      const tableName2 = 'Portfolio Results';
      const tableDomElementName2 = 'portfolioInfoTable2';
      this.renderTable(tableDomElementName2, tableName2, this.portfolioResultsData, columnsOrder2);
      // Access the rendered table element to modify cell values and styles
      const tableElement2 = document.getElementById(tableDomElementName2);
      if (tableElement2) {
        const rows = tableElement2.getElementsByTagName('tr');
        for (let i = 1; i < rows.length; i++) { // Start from 1 to skip the header row
            const cells = rows[i].getElementsByTagName('td');
            for (let j = 0; j < cells.length; j++) {
              //convert date values to ISO format and strip timestamp
              if (columnsOrder2[j] === 'From Date') {
                console.log('date:', cells[j].textContent);
                const date = new Date(cells[j].textContent || '');
                cells[j].textContent = date.toISOString().split('T')[0].replace(/-(\d{2})-/, (match, p1) => `-${new Date(date).toLocaleString('default', { month: 'short' })}-`);
              }
              if (columnsOrder2[j] === 'Total Value' || columnsOrder2[j] === 'Adjusted Total Value') {
                cells[j].textContent = `${currency}${(parseFloat(cells[j].textContent || '0')).toFixed(2)}`;
                cells[j].style.textAlign = 'right'; // Right justify the values
              }
            }
        }
      }

    }

    renderTable(tableElementId: string, tableName: string, tableData: any, columnsOrder: string[]): void {
       //Create a table to display the portfolio weights delta
      //The table should be built dynamically based on the columns in the data
      //The first row should be the column names
      const table1 = document.createElement('table');
      table1.classList.add('portfolio-info-table');
      const header1 = table1.createTHead();
      const headerRow1 = header1.insertRow(0);
      
      //Create the header row
      columnsOrder.forEach((key: any) => {
        const headerCell = headerRow1.insertCell();
        headerCell.textContent = key;
      });
      //Create a table body
      const body1 = table1.createTBody();
      tableData.forEach((row: any) => {
        const bodyRow = body1.insertRow();
        columnsOrder.forEach((key: any) => {
            const cell = bodyRow.insertCell();
              cell.textContent = row[key];
            }
        );
      }
      );

      // Style the table headers
      headerRow1.style.fontWeight = 'bold';
      headerRow1.style.border = '1px solid black'; // Add border to header row
      headerRow1.childNodes.forEach((cell: any) => {
        cell.style.border = '1px solid black'; // Add border to header cells
        cell.style.padding = '5px'; // Add padding to create a gap between the cell text and borders
      });
      // Style the table cells
      body1.childNodes.forEach((row: any) => {
        row.style.border = '1px solid black'; // Add border to table rows
        row.childNodes.forEach((cell: any, index: number) => {
          cell.style.border = '1px solid black'; // Add border to table cells
          cell.style.padding = '5px'; // Add padding to create a gap between the cell text and borders
        });
      });
      //Append the table to the DOM
      const tableTitle = document.createElement('h3');
      tableTitle.textContent = tableName;
      const portfolioWeightsDeltaDiv = document.getElementById(tableElementId);
      if (portfolioWeightsDeltaDiv) {
        portfolioWeightsDeltaDiv.innerHTML = '';
        portfolioWeightsDeltaDiv.appendChild(tableTitle);
        portfolioWeightsDeltaDiv.appendChild(table1);
      } else {
        console.error(tableElementId, 'is null');
      }
    }


    renderTimeSeriesPlot(mergedStockData: any): void {
      const uniqueTickers: string[] = [...new Set(this.mergedStockData.map((d: any) => d['Ticker'] as string))] as string[];
    
      //for each ticker create a trace object with x and y values
      const data1: Partial<Plotly.ScatterData>[] = [];
      for (let i = 0; i < uniqueTickers.length; i++) {
        const filteredData = this.mergedStockData.filter((d: any) => d['Ticker'] === uniqueTickers[i]);
        const trace: Partial<Plotly.ScatterData> = {
          x: filteredData.map((d: any) => new Date(d['Date']).toISOString().split('T')[0]),
          y: filteredData.map((d: any) => parseFloat(d['Adj Close'])),
          type: 'scatter' as Plotly.PlotType,
          name: uniqueTickers[i]
        };
        data1.push(trace);
      }

      console.log('Sample data for debugging:', data1);

      // Calculate the y-axis range
      const adjCloseValues = this.mergedStockData.map((d: any) => parseFloat(d['Adj Close']));
      const minY = Math.min(...adjCloseValues);
      const maxY = Math.max(...adjCloseValues);

      // Extract unique dates for x-axis ticks
      const uniqueDates = [...new Set(this.mergedStockData.map((d: any) => new Date(d['Start Date']).toISOString().split('T')[0]))];
      console.log(uniqueDates);
      
      // Increase y-axis granularity
      const yTickStep = (maxY - minY) / 20; // Increase the number of ticks
      const yTicks = Array.from({ length: 21 }, (_, i) => minY + i * yTickStep);
      const roundedYTicks = yTicks.map(tick => Math.round(tick / 10) * 10);

      const graphHeightPercentage = window.innerHeight * 0.5; // Set height as a % of the window height
      const graphWidthPercentage = window.innerWidth * 0.5; // Set width as a % of the window width

      const layout1: Partial<Plotly.Layout> = {
        title: {
          text: 'Ticker AdjClose Price Over Time',
        },
        yaxis2: {
          autorange: true,
          type: 'linear' as Plotly.AxisType,
          overlaying: 'y',
          side: 'right',
          title: {
            text: 'Portfolio Value',
            standoff: 20
          },
          tickfont: {
            size: 10
          },
        },
        xaxis: {
          range: [uniqueDates[0], uniqueDates[uniqueDates.length - 1]],
          tickvals: uniqueDates,
          type: 'date' as Plotly.AxisType,
          tickangle: -45, // Rotate the x-axis ticks from bottom to top but incline them by 45 degrees
            showline: true,
            showgrid: true,
            title: {
            text: 'Date',
            standoff: 20 // Add space between axis title and labels
            },
            tickfont: {
              size: 10 // Set font size of tick labels to 10
            },
            tickformat: '%Y-%m-%d', // Format x-axis labels as dates
          },
          yaxis: {
            autorange: true,
            type: 'linear' as Plotly.AxisType,
            showline: true,
            showgrid: true,
            title: {
            text: 'Adjusted Close Price',
            standoff: 20 // Add space between axis title and labels
            },
            tickfont: {
              size: 10 // Set font size of tick labels to 10
            },
          },
          legend: {
            title: {
            text: 'Tickers' // Legend title
            }
          },
        height: graphHeightPercentage, 
        width: graphWidthPercentage, 
      };

      Plotly.newPlot('tickerTimeSeriesPlotLeft', data1, layout1);
      Plotly.newPlot('tickerTimeSeriesPlotRight', data1, layout1);
    }

    renderDailyReturnsPlot(mergedStockData: any): void {
      const uniqueTickers: string[] = [...new Set(this.mergedStockData.map((d: any) => d['Ticker'] as string))] as string[];
    
      //for each ticker create a trace object with x and y values
      const data1: Partial<Plotly.ScatterData>[] = [];
      for (let i = 0; i < uniqueTickers.length; i++) {
        const filteredData = this.mergedStockData.filter((d: any) => d['Ticker'] === uniqueTickers[i]);
        const trace: Partial<Plotly.ScatterData> = {
          x: filteredData.map((d: any) => new Date(d['Date']).toISOString().split('T')[0]),
          y: filteredData.map((d: any) => parseFloat(d['Daily Return'])),
          type: 'scatter' as Plotly.PlotType,
          name: uniqueTickers[i]
        };
        data1.push(trace);
      }

      console.log('Sample daily returns data for debugging:', data1);

      // Calculate the y-axis range
      const dailyReturns = this.mergedStockData.map((d: any) => parseFloat(d['Daily Return']));
      const minY = Math.min(...dailyReturns);
      const maxY = Math.max(...dailyReturns);

      // Extract unique dates for x-axis ticks
      const uniqueDates = [...new Set(this.mergedStockData.map((d: any) => new Date(d['Start Date']).toISOString().split('T')[0]))];
      console.log(uniqueDates);
      
      // Increase y-axis granularity
      const yTickStep = (maxY - minY) / 20; // Increase the number of ticks
      const yTicks = Array.from({ length: 21 }, (_, i) => minY + i * yTickStep);
      const roundedYTicks = yTicks.map(tick => Math.round(tick / 10) * 10);

      const graphHeightPercentage = window.innerHeight * 0.5; // Set height as a % of the window height
      const graphWidthPercentage = window.innerWidth * 0.5; // Set width as a % of the window width

      const histogramData: Partial<Plotly.Data>[] = uniqueTickers.map(ticker => {
        const tickerData = this.mergedStockData.filter((d: any) => d['Ticker'] === ticker).map((d: any) => parseFloat(d['Daily Return']));
        return {
          x: tickerData,
          type: 'histogram' as Plotly.PlotType,
          name: ticker,
          opacity: 0.5,
        };
      });

      const histogramLayout: Partial<Plotly.Layout> = {
        title: {
        text: 'Histogram of Daily Returns',
        },
        xaxis: {
        title: {
          text: 'Daily Return',
        },
        },
        yaxis: {
        title: {
          text: 'Frequency',
        },
        },
        height: graphHeightPercentage, 
        width: graphWidthPercentage, 
      };

      Plotly.newPlot('tickerDailyReturnsPlotLeft', histogramData.flat(), histogramLayout);
      Plotly.newPlot('tickerDailyReturnsPlotRight', histogramData.flat(), histogramLayout);
    }

    renderPortfolioWeightAnalysisPlot(portfolioWeightsData: any, portfolioResultsData: any): void {

      //for each ticker create a trace object with x and y values
      const data1: Partial<Plotly.ScatterData>[] = [];
        const uniqueTickers: string[] = [...new Set(portfolioWeightsData.map((d: any) => d['Ticker'] as string))] as string[];
        uniqueTickers.forEach(ticker => {
          const tickerData = portfolioWeightsData.filter((d: any) => d['Ticker'] === ticker);
          const trace: Partial<Plotly.ScatterData> = {
            x: tickerData.map((d: any) => new Date(d['From Date']).toISOString().split('T')[0]),
            y: tickerData.map((d: any) => parseFloat(d['Weight'])),
            type: 'scatter' as Plotly.PlotType,
            name: ticker
          };
          data1.push(trace);
        });

        const portfolioValueTrace: Partial<Plotly.ScatterData> = {
          x: portfolioResultsData.map((d: any) => new Date(d['From Date']).toISOString().split('T')[0]),
          y: portfolioResultsData.map((d: any) => parseFloat(d['Total Value'])),
          type: 'scatter' as Plotly.PlotType,
          name: 'Portfolio Value',
          yaxis: 'y2',
          line: {
            color: 'black',
            width: 5
          }
        };

        data1.push(portfolioValueTrace);

        const adjPortfolioValueTrace: Partial<Plotly.ScatterData> = {
          x: portfolioResultsData.map((d: any) => new Date(d['From Date']).toISOString().split('T')[0]),
          y: portfolioResultsData.map((d: any) => parseFloat(d['Adjusted Total Value'])),
          type: 'scatter' as Plotly.PlotType,
          name: 'Adjusted Portfolio Value',
          yaxis: 'y2',
          line: {
            color: 'red',
            width: 5
          }
        };

        data1.push(adjPortfolioValueTrace);

        // Update traces to have dashed lines for ticker weights
        data1.forEach(trace => {
          if (trace.name == 'Portfolio Value') {
            trace.line = {
              dash: 'solid'
            };
          }
          else if (trace.name == 'Adjusted Portfolio Value') {
            trace.line = {
              dash: 'solid'
            };
          }
          else {
            trace.line = {
              dash: 'dash'
            };
          }
        });


        // Extract unique dates for x-axis ticks
        const uniqueDates = [...new Set(this.portfolioWeightsData.map((d: any) => new Date(d['From Date']).toISOString().split('T')[0]))];
        console.log(uniqueDates);

        const graphHeightPercentage = window.innerHeight * 1.0; // Set height as a % of the window height
        const graphWidthPercentage = window.innerWidth * 1.0; // Set width as a % of the window width
  
        const layout1: Partial<Plotly.Layout> = {
          title: {
            text: 'Portfolio Weights and Value Over Time',
          },
          xaxis: {
            range: [uniqueDates[0], uniqueDates[uniqueDates.length - 1]],
            tickvals: uniqueDates,
            type: 'date' as Plotly.AxisType,
            tickangle: -45,
            showline: true,
            showgrid: true,
            title: {
              text: 'From Date',
              standoff: 20
            },
            tickfont: {
              size: 10
            },
            tickformat: '%Y-%m-%d',
          },
          yaxis: {
            autorange: true,
            type: 'linear' as Plotly.AxisType,
            showline: true,
            showgrid: true,
            title: {
              text: 'Weight (%)',
              standoff: 20
            },
            tickfont: {
              size: 10
            },
          },
          yaxis2: {
            autorange: true,
            type: 'linear' as Plotly.AxisType,
            showline: true,
            overlaying: 'y',
            side: 'right',
            title: {
              text: 'Portfolio Value',
              standoff: 20
            },
            tickfont: {
              size: 10
            },
          },
          legend: {
            title: {
              text: 'Tickers'
            }
          },
          height: graphHeightPercentage,
          width: graphWidthPercentage,
        };

        Plotly.newPlot('portfolioAnalysisPlot', data1, layout1);

  }
}