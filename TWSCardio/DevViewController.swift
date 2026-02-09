//
//  DevViewController.swift
//  HeadPhoneLogger
//
//  Created by 王濡垚 on 2024/7/22.
//

import Foundation
import Combine
import UIKit
import Charts
import CoreMotion
import Starscream

extension CMLogItem {
    static let bootTime = Date(timeIntervalSinceNow: -ProcessInfo.processInfo.systemUptime)

    func startTime() -> Date {
        return CMLogItem.bootTime.addingTimeInterval(self.timestamp)
    }
    
    func toTimestamp() -> TimeInterval {
        return self.startTime().timeIntervalSince1970
    }
}

struct HeadPhoneMotionData: Codable {
        
    var rotationRateX: Double
    var rotationRateY: Double
    var rotationRateZ: Double
    var accelerationX: Double
    var accelerationY: Double
    var accelerationZ: Double
    var quaternionX: Double
    var quaternionY: Double
    var quaternionZ: Double
    var location: Int
    var timestamp: TimeInterval
    var updateCount: Int
}

class MotionDataManager: ObservableObject {
    
    @Published var motionData: CMDeviceMotion?
    
    private var motionManager: CMMotionManager
    var headphoneMotionManager: CMHeadphoneMotionManager

    init() {
        self.motionManager = CMMotionManager()
        self.headphoneMotionManager = CMHeadphoneMotionManager()
    }

    func startUpdates() {
        guard headphoneMotionManager.isDeviceMotionAvailable else {
            DispatchQueue.main.async {
                self.motionData = nil
                // Assuming you have a way to update textView outside of this class
                // Notify the view controller to update the textView
                NotificationCenter.default.post(name: Notification.Name("DeviceMotionNotSupported"), object: nil)
            }
            return
        }

        headphoneMotionManager.startDeviceMotionUpdates(to: OperationQueue.current!) { [weak self] motion, error in
            guard let self = self else { return }
            guard let motion = motion, error == nil else { return }
            DispatchQueue.main.async {
                self.motionData = motion
                self.printData(motion)
            }
        }
    }

    func stopUpdates() {
        headphoneMotionManager.stopDeviceMotionUpdates()
    }

    private func printData(_ data: CMDeviceMotion) {
        
        let timestamp = data.startTime()
        let timestampUnix = data.toTimestamp()
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        let formattedTimestamp = dateFormatter.string(from: timestamp)
        
//        print(data.toTimestamp(), formattedTimestamp)
    }
}

class TimeValueFormatter: IndexAxisValueFormatter {
    let dateFormatter: DateFormatter
    
    override init() {
        dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm:ss"
        super.init()
    }
    
    override func stringForValue(_ value: Double, axis: AxisBase?) -> String {
        let date = Date(timeIntervalSince1970: value)
        return dateFormatter.string(from: date)
    }
}


class DevViewController: UIViewController, WebSocketDelegate{
    
    let chartView = LineChartView()
    let accelerometerChartView = LineChartView()
    let socketChartView = LineChartView()
    let textView = UITextView()
    let socketTextView = UITextView()
    
    
    lazy var button: UIButton = {
        let button = UIButton(type: .system)
//        button.frame = CGRect(x: self.view.bounds.width / 4, y: self.view.bounds.maxY - 100,
//                              width: self.view.bounds.width / 2, height: 50)
        button.setTitle("Export to CSV",for: .normal)
        button.setTitleColor(.white, for: .normal)
        button.titleLabel?.font = UIFont.systemFont(ofSize: 15)
        button.layer.cornerRadius = 10
        button.backgroundColor = .systemBlue
        button.addTarget(self, action: #selector(Tap), for: .touchUpInside)
        
        return button
    }()
    
    let writer = CSVWriter()
    let f = DateFormatter()
    var write: Bool = false
    var fileUrl: URL!
    var motionDataManager = MotionDataManager()
    var cancellables: Set<AnyCancellable> = []
    
    var XdataEntries = [ChartDataEntry]()
    var YdataEntries = [ChartDataEntry]()
    var ZdataEntries = [ChartDataEntry]()
    var WdataEntries = [ChartDataEntry]()
    var XchartDataSet: LineChartDataSet!
    var YchartDataSet: LineChartDataSet!
    var ZchartDataSet: LineChartDataSet!
    var WchartDataSet: LineChartDataSet!
    
    var rotationRateXdataEntries = [ChartDataEntry]()
    var rotationRateYdataEntries = [ChartDataEntry]()
    var rotationRateZdataEntries = [ChartDataEntry]()
    var rotationRateXchartDataSet: LineChartDataSet!
    var rotationRateYchartDataSet: LineChartDataSet!
    var rotationRateZchartDataSet: LineChartDataSet!
    
    var accelerationXdataEntries = [ChartDataEntry]()
    var accelerationYdataEntries = [ChartDataEntry]()
    var accelerationZdataEntries = [ChartDataEntry]()
    var accelerationXchartDataSet: LineChartDataSet!
    var accelerationYchartDataSet: LineChartDataSet!
    var accelerationZchartDataSet: LineChartDataSet!
    
    var socketDataEntries = [ChartDataEntry]()
    var socketDataChartDataSet: LineChartDataSet!
    
    var xValue: Double = 0
    var newDataPointCounter = 0
    
    var socket: WebSocket!
    var ipTextField: UITextField!
    var portTextField: UITextField!
    var connectButton: UIButton!
    var isConnected = false
    var reconnectInterval: TimeInterval = 5
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupWebSocket()
        setupViews()
        setupInitialDataEntries()
        setupChartData()
        motionDataManager.$motionData
            .receive(on: RunLoop.main)
            .sink{ [weak self] motionData in
                self?.updateView(with: motionData)
                self?.writeMotionDataToCSV(with: motionData)
            }
            .store(in: &cancellables)
        motionDataManager.startUpdates()
    }
    
    func setupWebSocket() {
        var request = URLRequest(url: URL(string: "ws://192.168.31.205:8765")!) // Change it to your socket address 192.168.31.205
        request.timeoutInterval = 5
        socket = WebSocket(request: request)
        socket.delegate = self
        socket.connect()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        DispatchQueue.main.async {
            self.ipTextField.becomeFirstResponder()
            self.ipTextField.resignFirstResponder()
            
            self.portTextField.becomeFirstResponder()
            self.portTextField.resignFirstResponder()
        }
    }
    
    func setupViews() {
            view.backgroundColor = .systemBackground
        
            let scrollView = UIScrollView()
            let contentView = UIView()
            
            scrollView.translatesAutoresizingMaskIntoConstraints = false
            contentView.translatesAutoresizingMaskIntoConstraints = false
            
            view.addSubview(scrollView)
            scrollView.addSubview(contentView)
        
            // Setup chartView
            chartView.translatesAutoresizingMaskIntoConstraints = false
            chartView.backgroundColor = .systemBackground
            chartView.drawGridBackgroundEnabled = false
            chartView.drawBordersEnabled = true
            chartView.borderColor = .gray
            chartView.borderLineWidth = 1.0
            

            // X-Axis settings
            let xAxis = chartView.xAxis
            xAxis.drawGridLinesEnabled = true // 启用 X 轴网格线
            xAxis.gridColor = .gray // 设置网格线颜色
            xAxis.gridLineWidth = 1.0 // 设置网格线宽度
            xAxis.valueFormatter = TimeValueFormatter()
            xAxis.labelCount = 4

            // Y-Axis settings (left and right)
            let axisSettings: (AxisBase) -> Void = { axis in
                axis.drawGridLinesEnabled = true // 启用 Y 轴网格线
                axis.gridColor = .gray // 设置网格线颜色
                axis.gridLineWidth = 1.0 // 设置网格线宽度
            }

            axisSettings(chartView.leftAxis)
            chartView.rightAxis.enabled = false
        
            accelerometerChartView.translatesAutoresizingMaskIntoConstraints = false
            accelerometerChartView.backgroundColor = .systemBackground
            accelerometerChartView.drawGridBackgroundEnabled = false
            accelerometerChartView.drawBordersEnabled = true
            accelerometerChartView.borderColor = .gray
            accelerometerChartView.borderLineWidth = 1.0

            let accelerometer_xAxis = accelerometerChartView.xAxis
            accelerometer_xAxis.drawGridLinesEnabled = true // 启用 X 轴网格线
            accelerometer_xAxis.gridColor = .gray // 设置网格线颜色
            accelerometer_xAxis.gridLineWidth = 1.0 // 设置网格线宽度
            accelerometer_xAxis.valueFormatter = TimeValueFormatter()
            accelerometer_xAxis.labelCount = 4

            axisSettings(accelerometerChartView.leftAxis)
            //  axisSettings(accelerometerChartView.rightAxis)
            accelerometerChartView.rightAxis.enabled = false
        
        
            socketChartView.translatesAutoresizingMaskIntoConstraints = false
            socketChartView.backgroundColor = .systemBackground
            socketChartView.drawGridBackgroundEnabled = false
            socketChartView.drawBordersEnabled = true
            socketChartView.borderColor = .gray
            socketChartView.borderLineWidth = 1.0
            
    //            accelerometerChartView.heightAnchor.constraint(equalToConstant: 100).isActive = true

            let socket_xAxis = socketChartView.xAxis
            socket_xAxis.drawGridLinesEnabled = true // 启用 X 轴网格线
            socket_xAxis.gridColor = .gray // 设置网格线颜色
            socket_xAxis.gridLineWidth = 1.0 // 设置网格线宽度
            socket_xAxis.valueFormatter = TimeValueFormatter()
            socket_xAxis.labelCount = 4

            axisSettings(socketChartView.leftAxis)
            //            axisSettings(socketChartView.rightAxis)
            socketChartView.rightAxis.enabled = false
            
            // Setup textView
            textView.backgroundColor = .systemBackground
            textView.textColor = .gray
            textView.translatesAutoresizingMaskIntoConstraints = false
            textView.isEditable = false
            textView.text = "Motion Data will be displayed here."

            // Setup IP TextField
            ipTextField = createTextField(placeholder: "Enter IP Address")

            // Setup Port TextField
            portTextField = createTextField(placeholder: "Enter Port")
            portTextField.keyboardType = .numberPad

            // Setup Connect Button
            connectButton = createButton(title: "Connect", action: #selector(connectButtonTapped))

            socketTextView.backgroundColor = .systemBackground
            socketTextView.textColor = .gray
            socketTextView.translatesAutoresizingMaskIntoConstraints = false
            socketTextView.isEditable = false
            socketTextView.text = "socket data details will be displayed here."
        
            // Setup Start Button
            button = createButton(title: "Export to CSV", action: #selector(Tap))
            
            let chartViewTitle = createTitleLabel(text: "Gyro Raw Data", fontSize: 16)
            let accelerometerChartViewTitle = createTitleLabel(text: "Accelerometer Data", fontSize: 16)
            let socketChartViewTitle = createTitleLabel(text: "Reconstructed BCG", fontSize: 16)
        

            chartView.translatesAutoresizingMaskIntoConstraints = false
            ipTextField.translatesAutoresizingMaskIntoConstraints = false
            portTextField.translatesAutoresizingMaskIntoConstraints = false
            connectButton.translatesAutoresizingMaskIntoConstraints = false
            textView.translatesAutoresizingMaskIntoConstraints = false
            socketTextView.translatesAutoresizingMaskIntoConstraints = false
            button.translatesAutoresizingMaskIntoConstraints = false
            
            chartViewTitle.translatesAutoresizingMaskIntoConstraints = false
            accelerometerChartViewTitle.translatesAutoresizingMaskIntoConstraints = false
            socketChartViewTitle.translatesAutoresizingMaskIntoConstraints = false

            NSLayoutConstraint.activate([
                // ScrollView Constraints
                scrollView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
                scrollView.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor),
                scrollView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor),
                scrollView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
                
                // ContentView Constraints
                contentView.topAnchor.constraint(equalTo: scrollView.topAnchor),
                contentView.leadingAnchor.constraint(equalTo: scrollView.leadingAnchor),
                contentView.trailingAnchor.constraint(equalTo: scrollView.trailingAnchor),
                contentView.bottomAnchor.constraint(equalTo: scrollView.bottomAnchor),
                contentView.widthAnchor.constraint(equalTo: scrollView.widthAnchor) // 设置contentView的宽度与scrollView相同
            ])
            
            contentView.addSubview(chartViewTitle)
            contentView.addSubview(chartView)
            contentView.addSubview(accelerometerChartViewTitle)
            contentView.addSubview(accelerometerChartView)
            contentView.addSubview(socketChartViewTitle)
            contentView.addSubview(socketChartView)
            //            contentView.addSubview(ipTextField)
            //            contentView.addSubview(portTextField)
            //            contentView.addSubview(connectButton)
            //            contentView.addSubview(textView)
            //            contentView.addSubview(socketTextView)
            contentView.addSubview(button)
            
            NSLayoutConstraint.activate([
                // Chart View Constraints
                
                // Chart View Title Constraints
                chartViewTitle.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 4),
                chartViewTitle.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
                chartViewTitle.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
                chartViewTitle.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),

                // Chart View Constraints
                chartView.topAnchor.constraint(equalTo: chartViewTitle.bottomAnchor, constant: 4),
                chartView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 4),
                chartView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -4),
                chartView.heightAnchor.constraint(equalToConstant: 180),

                // Accelerometer Chart View Title Constraints
                accelerometerChartViewTitle.topAnchor.constraint(equalTo: chartView.bottomAnchor, constant: 4),
                accelerometerChartViewTitle.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
                accelerometerChartViewTitle.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
                accelerometerChartViewTitle.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),

                // Accelerometer Chart View Constraints
                accelerometerChartView.topAnchor.constraint(equalTo: accelerometerChartViewTitle.bottomAnchor, constant: 4),
                accelerometerChartView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 4),
                accelerometerChartView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -4),
                accelerometerChartView.heightAnchor.constraint(equalToConstant: 180),

                // Socket Chart View Title Constraints
                socketChartViewTitle.topAnchor.constraint(equalTo: accelerometerChartView.bottomAnchor, constant: 4),
                socketChartViewTitle.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
                socketChartViewTitle.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
                socketChartViewTitle.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),

                // Socket Chart View Constraints
                socketChartView.topAnchor.constraint(equalTo: socketChartViewTitle.bottomAnchor, constant: 4),
                socketChartView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 4),
                socketChartView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -4),
                socketChartView.heightAnchor.constraint(equalToConstant: 180),

                //                // IP TextField Constraints
                //                ipTextField.topAnchor.constraint(equalTo: chartView.bottomAnchor, constant: 20),
                //                ipTextField.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
                //                ipTextField.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
                //                ipTextField.heightAnchor.constraint(equalToConstant: 40),
                //
                //                // Port TextField Constraints
                //                portTextField.topAnchor.constraint(equalTo: ipTextField.bottomAnchor, constant: 20),
                //                portTextField.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
                //                portTextField.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
                //                portTextField.heightAnchor.constraint(equalToConstant: 40),

                // Connect Button Constraints
                //                connectButton.centerXAnchor.constraint(equalTo: portTextField.centerXAnchor),
                //                connectButton.topAnchor.constraint(equalTo: chartView.bottomAnchor, constant: 20),
                //                connectButton.widthAnchor.constraint(equalTo: contentView.widthAnchor, multiplier: 0.5),
                //                connectButton.heightAnchor.constraint(equalToConstant: 50),

                // Text View Constraints
                //                textView.topAnchor.constraint(equalTo: socketChartView.bottomAnchor, constant: 20),
                //                textView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
                //                textView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
                //                textView.heightAnchor.constraint(equalToConstant: 100),

                // Socket Text View Constraints
                //                socketTextView.topAnchor.constraint(equalTo: socketChartView.bottomAnchor, constant: 20),
                //                socketTextView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
                //                socketTextView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
                //                socketTextView.heightAnchor.constraint(equalToConstant: 100),

                // Start Button Constraints
                // button.topAnchor.constraint(equalTo: socketChartView.bottomAnchor, constant: 20),
                //                button.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
                //                button.widthAnchor.constraint(equalTo: contentView.widthAnchor, multiplier: 0.5),
                //                button.heightAnchor.constraint(equalToConstant: 50),
                // button.bottomAnchor.constraint(equalTo: contentView.safeAreaLayoutGuide.bottomAnchor, constant: -20)
                // button.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -20)
                
                // button.topAnchor.constraint(equalTo: socketChartView.bottomAnchor, constant: 20),
                button.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
                button.widthAnchor.constraint(equalTo: contentView.widthAnchor, multiplier: 0.4),
                button.heightAnchor.constraint(equalToConstant: 40),
                button.bottomAnchor.constraint(equalTo:  view.safeAreaLayoutGuide.bottomAnchor, constant: -1)
                
            ])

            DispatchQueue.main.async {
                self.ipTextField.becomeFirstResponder()
                self.ipTextField.resignFirstResponder()
                
                self.portTextField.becomeFirstResponder()
                self.portTextField.resignFirstResponder()
            }

        }
    
    func createTitleLabel(text: String, fontSize: CGFloat) -> UILabel {
        let titleLabel = UILabel()
        titleLabel.text = text
        titleLabel.font = .systemFont(ofSize: fontSize)
        titleLabel.textColor = .black
        titleLabel.textAlignment = .center
        return titleLabel
    }
    
    func createTextField(placeholder: String) -> UITextField {
        let textField = UITextField()
        textField.placeholder = placeholder
        textField.borderStyle = .roundedRect
        textField.translatesAutoresizingMaskIntoConstraints = false
        
        // 创建工具栏
        let toolbar = UIToolbar()
        toolbar.sizeToFit()
        
        // 创建完成按钮
        let doneButton = UIBarButtonItem(barButtonSystemItem: .done, target: self, action: #selector(doneButtonTapped))
        let flexSpace = UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil)
        
        // 将按钮添加到工具栏
        toolbar.items = [flexSpace, doneButton]
        
        // 设置工具栏作为输入附件视图
        textField.inputAccessoryView = toolbar
        
        return textField
    }
    
    @objc func doneButtonTapped() {
        view.endEditing(true) // 关闭键盘
    }

    private func createButton(title: String, action: Selector) -> UIButton {
        let button = UIButton(type: .system)
        button.setTitle(title, for: .normal)
        button.translatesAutoresizingMaskIntoConstraints = false
        button.addTarget(self, action: action, for: .touchUpInside)
        button.setTitleColor(.white, for: .normal)
        button.titleLabel?.font = UIFont.systemFont(ofSize: 15)
        button.layer.cornerRadius = 10
        button.backgroundColor = .systemBlue
        return button
    }

    @objc func connectButtonTapped() {
        Task {
            await connectToWebSocket()
        }
    }
    
    func connectToWebSocket() async {
            guard let ip = ipTextField.text, !ip.isEmpty,
                  let port = portTextField.text, !port.isEmpty else {
                // Show error message
                return
            }
            
            let urlString = "ws://\(ip):\(port)"
            guard let url = URL(string: urlString) else {
                // Show error message
                return
            }
            
            await withCheckedContinuation { continuation in
                var request = URLRequest(url: url)
                request.timeoutInterval = 5
                socket = WebSocket(request: request)
                socket.delegate = self
                socket.connect()
                
                // Delay to simulate async connection
                DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
                    continuation.resume()
                }
            }
        }
    
    
    func setupInitialDataEntries() {
    }
    
    func setupChartData() {
        rotationRateXchartDataSet = createDataSet(entries: rotationRateXdataEntries, label: "rotation x", color: NSUIColor.red)
        rotationRateYchartDataSet = createDataSet(entries: rotationRateYdataEntries, label: "rotation y", color: NSUIColor.blue)
        rotationRateZchartDataSet = createDataSet(entries: rotationRateZdataEntries, label: "rotation z", color: NSUIColor.green)
        accelerationXchartDataSet = createDataSet(entries: accelerationXdataEntries, label: "acceleration x", color: NSUIColor.red)
        accelerationYchartDataSet = createDataSet(entries: accelerationYdataEntries, label: "acceleration y", color: NSUIColor.blue)
        accelerationZchartDataSet = createDataSet(entries: accelerationZdataEntries, label: "acceleration z", color: NSUIColor.green)
        
        socketDataChartDataSet = createDataSet(entries: socketDataEntries, label: "Reconstructed SCG", color: NSUIColor.red)
        
        let chartData = LineChartData(dataSets: [
            rotationRateXchartDataSet,
            rotationRateYchartDataSet,
            rotationRateZchartDataSet,
        ])
        
        chartView.data = chartData
        chartView.xAxis.labelPosition = .bottom
        
        
        let acceleration_chartData = LineChartData(dataSets: [
            accelerationXchartDataSet,
            accelerationYchartDataSet,
            accelerationZchartDataSet
        ])
        
        accelerometerChartView.data = acceleration_chartData
        accelerometerChartView.xAxis.labelPosition = .bottom
        
        let socket_chartData = LineChartData(
            dataSets: [socketDataChartDataSet]
        )
        socketChartView.data = socket_chartData
        socketChartView.xAxis.labelPosition = .bottom
    }
    
    func createDataSet(entries: [ChartDataEntry], label: String, color: NSUIColor) -> LineChartDataSet {
        let dataSet = LineChartDataSet(entries: entries, label: label)
        dataSet.drawCirclesEnabled = false
        dataSet.setColor(color)
        dataSet.drawValuesEnabled = false
        dataSet.mode = .linear
        dataSet.lineWidth = 2.5
        return dataSet
    }
    
    @objc func didUpdateChartView(with motionData: CMDeviceMotion?) {
        guard let motionData = motionData else {
            return
        }
        
        xValue += 1
        
        let rotationRateXDataEntry = ChartDataEntry(x: motionData.timestamp, y: motionData.rotationRate.x)
        let rotationRateYDataEntry = ChartDataEntry(x: motionData.timestamp, y: motionData.rotationRate.y)
        let rotationRateZDataEntry = ChartDataEntry(x: motionData.timestamp, y: motionData.rotationRate.z)
        
        let accelerationXDataEntry = ChartDataEntry(x: motionData.timestamp, y: motionData.userAcceleration.x)
        let accelerationYDataEntry = ChartDataEntry(x: motionData.timestamp, y: motionData.userAcceleration.y)
        let accelerationZDataEntry = ChartDataEntry(x: motionData.timestamp, y: motionData.userAcceleration.z)
        
        // Update chart and text view on the main thread
        DispatchQueue.main.async {
            self.updateView(with: motionData)
        }
        
//        sendMessage(String(motionData.attitude.quaternion.x))
    }

    func updateChartView(rotationRatexDataEntry: ChartDataEntry, rotationRateyDataEntry: ChartDataEntry, rotationRatezDataEntry: ChartDataEntry,
                         accelerationxDataEntry: ChartDataEntry, accelerationyDataEntry: ChartDataEntry, accelerationzDataEntry: ChartDataEntry) {
        let maxEntries = 150
        
        xValue += 1
        // Remove old entries if needed
        removeOldEntriesIfNeeded(maxEntries: maxEntries)
        
        // Add new entries on the main thread
        DispatchQueue.main.async {
            self.rotationRateXdataEntries = self.appendNewEntry(rotationRatexDataEntry, to: self.rotationRateXdataEntries, in: self.rotationRateXchartDataSet)
            self.rotationRateYdataEntries = self.appendNewEntry(rotationRateyDataEntry, to: self.rotationRateYdataEntries, in: self.rotationRateYchartDataSet)
            self.rotationRateZdataEntries = self.appendNewEntry(rotationRatezDataEntry, to: self.rotationRateZdataEntries, in: self.rotationRateZchartDataSet)
            self.accelerationXdataEntries = self.appendNewEntry(accelerationxDataEntry, to: self.accelerationXdataEntries, in: self.accelerationXchartDataSet)
            self.accelerationYdataEntries = self.appendNewEntry(accelerationyDataEntry, to: self.accelerationYdataEntries, in: self.accelerationYchartDataSet)
            self.accelerationZdataEntries = self.appendNewEntry(accelerationzDataEntry, to: self.accelerationZdataEntries, in: self.accelerationZchartDataSet)
            
            self.chartView.data?.notifyDataChanged()
            self.chartView.notifyDataSetChanged()
            self.chartView.moveViewToX(rotationRatexDataEntry.x)
            
            
            self.accelerometerChartView.data?.notifyDataChanged()
            self.accelerometerChartView.notifyDataSetChanged()
            self.accelerometerChartView.moveViewToX(rotationRatexDataEntry.x)
        }
//        print(self.accelerationXdataEntries.first, xValue)
    }

    func removeOldEntriesIfNeeded(maxEntries: Int) {
        if rotationRateXdataEntries.count > maxEntries {
            rotationRateXdataEntries = removeOldEntry(from: rotationRateXdataEntries, in: rotationRateXchartDataSet)
            rotationRateYdataEntries = removeOldEntry(from: rotationRateYdataEntries, in: rotationRateYchartDataSet)
            rotationRateZdataEntries = removeOldEntry(from: rotationRateZdataEntries, in: rotationRateZchartDataSet)
            accelerationXdataEntries = removeOldEntry(from: accelerationXdataEntries, in: accelerationXchartDataSet)
            accelerationYdataEntries = removeOldEntry(from: accelerationYdataEntries, in: accelerationYchartDataSet)
            accelerationZdataEntries = removeOldEntry(from: accelerationZdataEntries, in: accelerationZchartDataSet)
        }
    }

    func removeOldEntry(from dataEntries: [ChartDataEntry], in dataSet: LineChartDataSet) -> [ChartDataEntry] {
        var updatedEntries = dataEntries
        if !updatedEntries.isEmpty {
            dataSet.removeEntry(updatedEntries.first!)
            updatedEntries.removeFirst()
        }
        return updatedEntries
    }

    func appendNewEntry(_ dataEntry: ChartDataEntry, to dataEntries: [ChartDataEntry], in dataSet: LineChartDataSet) -> [ChartDataEntry] {
        var updatedEntries = dataEntries
        updatedEntries.append(dataEntry)
        dataSet.append(dataEntry)
        return updatedEntries
    }

    func updateTextView(with motionData: CMDeviceMotion?) {
        guard let motionData = motionData else {
            return
        }
        
        let locationDescription: String

        switch motionData.sensorLocation.rawValue {
        case 1:
            locationDescription = "left"
        case 2:
            locationDescription = "right"
        default:
            locationDescription = "error"
        }
        
        self.textView.text = """
            Rotation Rate:
                x: \(motionData.rotationRate.x)
                y: \(motionData.rotationRate.y)
                z: \(motionData.rotationRate.z)
            Acceleration:
                x: \(motionData.userAcceleration.x)
                y: \(motionData.userAcceleration.y)
                z: \(motionData.userAcceleration.z)
            Location:
                   \(locationDescription)
        """
    }

    func updateView(with motionData: CMDeviceMotion?) {
        
        //        updateTextView(with: motionData)
        
        guard let motionData = motionData else{
            return
        }
        
        let timestamp = motionData.startTime()
        let timestampUnix = motionData.toTimestamp()
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm:ss.SSS"
        let formattedTimestamp = dateFormatter.string(from: timestamp)
        
//        print(motionData.toTimestamp(), motionData.timestamp, formattedTimestamp)
        
        let rotationRateXDataEntry = ChartDataEntry(x: motionData.toTimestamp(), y: motionData.rotationRate.x ?? 0)
        let rotationRateYDataEntry = ChartDataEntry(x: motionData.toTimestamp(), y: motionData.rotationRate.y ?? 0)
        let rotationRateZDataEntry = ChartDataEntry(x: motionData.toTimestamp(), y: motionData.rotationRate.z ?? 0)
        
        let accelerationXDataEntry = ChartDataEntry(x: motionData.toTimestamp(), y: motionData.userAcceleration.x ?? 0)
        let accelerationYDataEntry = ChartDataEntry(x: motionData.toTimestamp(), y: motionData.userAcceleration.y ?? 0)
        let accelerationZDataEntry = ChartDataEntry(x: motionData.toTimestamp(), y: motionData.userAcceleration.z ?? 0)
        
        sendMotionData(with: motionData)
        updateChartView(rotationRatexDataEntry: rotationRateXDataEntry, rotationRateyDataEntry: rotationRateYDataEntry, rotationRatezDataEntry: rotationRateZDataEntry,
                        accelerationxDataEntry: accelerationXDataEntry, accelerationyDataEntry: accelerationYDataEntry, accelerationzDataEntry: accelerationZDataEntry)
    }
    
    func updateSocketView(socketData: String?) {
        guard let socketData = socketData else {
            return
        }
        
        let maxEntries = 600
        let updateFrequency = 5 // 每5个数据点更新一次图表
        
        // 1. Extract timestamp and data from the socketData
        let result = extractTimestampAndData(from: socketData)
        let socketDataEntry = ChartDataEntry(x: result.timestamp, y: result.data)
        
        // 2. Append the new entry and remove old entries if needed
        self.socketDataEntries = self.appendNewEntry(socketDataEntry, to: self.socketDataEntries, in: self.socketDataChartDataSet)
        newDataPointCounter += 1
        
        if self.socketDataEntries.count > maxEntries {
            self.socketDataEntries = removeOldEntry(from: socketDataEntries, in: socketDataChartDataSet)
        }
        
        // 4. Update the chart view at the specified frequency
        if newDataPointCounter % updateFrequency == 0 {
            self.normalizeDataSet(self.socketDataChartDataSet, maxEntries: maxEntries)
            DispatchQueue.main.async {
                let socketChartData = LineChartData(dataSet: self.socketDataChartDataSet)
                self.socketChartView.data = socketChartData
                self.socketChartView.data?.notifyDataChanged()
                self.socketChartView.notifyDataSetChanged()
                self.socketChartView.moveViewToX(socketDataEntry.x)
            }
            newDataPointCounter = 0 // 重置计数器
        }
    }

    func normalizeDataSet(_ dataSet: LineChartDataSet, maxEntries: Int) {
        guard !dataSet.entries.isEmpty else { return }

        // Step 1: Extract values from the data set
        let values = dataSet.entries.map { $0.y }
        
        // Step 2: Calculate min and max values
        guard let minValue = values.min(), let maxValue = values.max(), minValue != maxValue else {
            // If all values are the same or no values, return
            return
        }
        
        // Step 3: Normalize the values to the range [-1, 1]
        let normalizedEntries = dataSet.entries.map { entry -> ChartDataEntry in
            let normalizedY = ((entry.y - minValue) / (maxValue - minValue)) * 2 - 1
            return ChartDataEntry(x: entry.x, y: normalizedY)
        }
        
        // Step 4: Limit the number of entries to maxEntries
        let limitedEntries = Array(normalizedEntries.suffix(maxEntries))
        
        // Step 5: Update the data set with normalized values
        dataSet.replaceEntries(limitedEntries)
    }

    
    func writeMotionDataToCSV(with motionData: CMDeviceMotion?){
        guard let motionData = motionData else{
            return
        }
        
        if write {
//            print("Write fileUrl", fileUrl)
            self.writer.write(motionData)
        }
        
    }
    
    @objc func Tap() {
        if write {
            write.toggle()
            writer.close()
            button.setTitle("Start", for: .normal)
            AlertView.action(self, handler: {[weak self](_) in self?.viewCreatedFiles()}, animated: true)
        } else {
            guard motionDataManager.headphoneMotionManager.isDeviceMotionAvailable else {
                AlertView.alert(self, "Sorry", "Your device is not supported.")
                return
            }
            write.toggle()
            button.setTitle("Stop", for: .normal)
            let dir = FileManager.default.urls(
                for: .documentDirectory,
                in: .userDomainMask
            ).first!

            let formatter = DateFormatter()
            formatter.dateFormat = "yyyyMMdd_HHmm"
            let filename = formatter.string(from: Date()) + "_motion.csv"
            fileUrl = dir.appendingPathComponent(filename)
            print("Tap fileUrl", fileUrl)

            do {
                // 创建空文件以确保路径存在
                FileManager.default.createFile(atPath: fileUrl.path, contents: nil, attributes: nil)
                // 打开文件
                writer.open(fileUrl)
            } catch {
                print("Error creating or opening file: \(error)")
            }
        }
    }
    
    func viewCreatedFiles() {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!

        // 获取正确的文件URL
        if let components = NSURLComponents(url: dir, resolvingAgainstBaseURL: true) {
            components.scheme = "shareddocuments"
            if let sharedDocuments = components.url {
//                print(sharedDocuments)
                UIApplication.shared.open(sharedDocuments, options: [:], completionHandler: nil)
            } else {
                AlertView.warning(self)
            }
        }
    }
    
    func extractTimestampAndData(from optionalJSONString: String?) -> (timestamp: Double, data: Double) {
        // Step 1: Ensure the string is not nil
        guard let jsonString = optionalJSONString else {
            return (0, 0) // Return default values if the string is nil
        }
        
        // Step 2: Convert the JSON string to Data
        guard let jsonData = jsonString.data(using: .utf8) else {
            return (0, 0) // Return default values if conversion to Data fails
        }
        
        // Step 3: Parse the JSON data into a dictionary
        do {
            if let dictionary = try JSONSerialization.jsonObject(with: jsonData, options: []) as? [String: Any] {
                // Step 4: Extract the timestamp, default to 0 if not available
                let timestamp = dictionary["timestamp"] as? Double ?? 0
                
                // Step 5: Extract the data (specifically "accelerationX"), default to 0 if not available
                let data = dictionary["bcg_signal"] as? Double ?? 0
                print(timestamp, data)
                return (timestamp, data)
            }
        } catch {
            print("Failed to parse JSON: \(error)")
            return (0, 0) // Return default values if parsing fails
        }
        
        // Return default values if the dictionary conversion fails
        return (0, 0)
    }
    
    func didReceive(event: Starscream.WebSocketEvent, client: any Starscream.WebSocketClient) {
        switch event {
        case .connected(let headers):
            isConnected = true
            print("WebSocket is connected: \(headers)")
            // WebSocket 连接成功后，可以发送数据
            // sendMessage("Hello, server!")
        case .disconnected(let reason, let code):
            isConnected = false
            print("WebSocket is disconnected: \(reason) with code: \(code)")
            // 尝试重连
            reconnect()
        case .text(let string):
            updateSocketView(socketData: string)
            socketTextView.text = string
            // 处理接收到的文本消息
        case .binary(let data):
            print("Received data: \(data.count) bytes")
            // 处理接收到的二进制数据
        case .ping(let pingData):
            print("Ping received")
        case .pong(let pongData):
            print("Pong received")
        case .viabilityChanged(let viability):
            print("Viability changed: \(viability)")
        case .reconnectSuggested(let reconnect):
            print("Reconnect suggested: \(reconnect)")
        case .cancelled:
            isConnected = false
            print("WebSocket is cancelled")
        case .error(let error):
            handleError(error)
        case .peerClosed:
            isConnected = false
            self.socketTextView.text = "websocket disconnected"
            print("WebSocket peer is closed")
            // 尝试重连
            reconnect()
        }
    }

    func sendMessage(_ message: String) {
        if isConnected {
            socket.write(string: message)
        }
    }
    
    func sendMotionData(with motionData: CMDeviceMotion?) {
        
        guard let motionData = motionData else{
            return
        }
        
        let headPhoneMotionData = HeadPhoneMotionData(
            rotationRateX: motionData.rotationRate.x,
            rotationRateY: motionData.rotationRate.y,
            rotationRateZ: motionData.rotationRate.z,
            accelerationX: motionData.userAcceleration.x,
            accelerationY: motionData.userAcceleration.y,
            accelerationZ: motionData.userAcceleration.z,
            quaternionX: motionData.attitude.quaternion.x,
            quaternionY: motionData.attitude.quaternion.y,
            quaternionZ: motionData.attitude.quaternion.z,
            location: motionData.sensorLocation.rawValue,
            timestamp: motionData.toTimestamp(),
            updateCount: Int(xValue)
        )
        
        let headPhoneMotionDataString = convertToJSONString(from: headPhoneMotionData)
        
        guard let headPhoneMotionDataString = headPhoneMotionDataString else{
            return
        }
        
        if isConnected {
            socket.write(string: headPhoneMotionDataString)
        }
    }

    func handleError(_ error: Error?) {
        if let e = error as? WSError {
            print("WebSocket encountered an error: \(e.message)")
        } else if let e = error {
            print("WebSocket encountered an error: \(e.localizedDescription)")
        } else {
            print("WebSocket encountered an error")
        }
    }

    @objc func sendButtonTapped(_ sender: UIButton) {
        sendMessage("Hello again, server!")
    }
    
    func reconnect() {
        guard !isConnected else { return }
        print("Attempting to reconnect in \(reconnectInterval) seconds...")
        DispatchQueue.main.asyncAfter(deadline: .now() + reconnectInterval) { [weak self] in
            guard let self = self else { return }
            self.setupWebSocket()
        }
    }
    
    func convertToJSONString(from data: HeadPhoneMotionData) -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        
        do {
            let jsonData = try encoder.encode(data)
            return String(data: jsonData, encoding: .utf8)
        } catch {
            print("Error encoding data: \(error.localizedDescription)")
            return nil
        }
    }
}



