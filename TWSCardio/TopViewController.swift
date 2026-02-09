//
//  TopViewController.swift
//  EarCardioApplicationDemo
//
//  Created by 王濡垚 on 2024/7/15.
//

import Foundation
import UIKit

class TopViewController: UIViewController, UITableViewDataSource, UITableViewDelegate {
    
    private lazy var table: UITableView = {
        print("TopView init")
        let table = UITableView(frame: self.view.bounds, style: .plain)
        table.autoresizingMask = [
          .flexibleWidth,
          .flexibleHeight
        ]
        table.rowHeight = 60
        table.translatesAutoresizingMaskIntoConstraints = false
        return table
    }()
    private var items: [UIViewController] = [DevViewController()]
    private var itemTitle: [String] =  ["Dev View"]
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.title = "TWSCardio"
        
        table.dataSource = self
        table.delegate = self
        view.addSubview(table)

    }
    
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return items.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = UITableViewCell(style: .default, reuseIdentifier: "cell")
        cell.textLabel?.text = itemTitle[indexPath.row]
        return cell
    }
    
    
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
        self.navigationController?.pushViewController(items[indexPath.row], animated: true)
    }
}


//#Preview{
//    TopViewController()
//}

