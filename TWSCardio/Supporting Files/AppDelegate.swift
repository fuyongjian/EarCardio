//
//  AppDelegate.swift
//  EarCardioApplicationDemo
//  Created by 王濡垚 on 2024/7/15.
//

import UIKit

@main
class AppDelegate: UIResponder, UIApplicationDelegate {
    
    // Remove the window property from AppDelegate if using SceneDelegate
    // var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        // Override point for customization after application launch.
        print("Override point for customization after application launch.")
        return true
    }

    // MARK: UISceneSession Lifecycle

    func application(_ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession, options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        print("load UI delegate")
        // Called when a new scene session is being created.
        // Use this method to select a configuration to create the new scene with.
        return UISceneConfiguration(name: "Default Configuration", sessionRole: connectingSceneSession.role)
    }

    func application(_ application: UIApplication, didDiscardSceneSessions sceneSessions: Set<UISceneSession>) {
        // Called when the user discards a scene session.
        // If any sessions were discarded while the application was not running, this will be called shortly after application:didFinishLaunchingWithOptions.
        // Use this method to release any resources that were specific to the discarded scenes, as they will not return.
    }
}

//@main
//class AppDelegate: UIResponder, UIApplicationDelegate {
//
//    var window: UIWindow?
//
//    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
//
//        // Create a new UIWindow instance with the screen bounds
//        window = UIWindow(frame: UIScreen.main.bounds)
//        guard let window = window else { return false }
//
//        // Create the root view controller
//        let topViewController = TopViewController()
//
//        // Embed the root view controller in a navigation controller
//        let navigationController = UINavigationController(rootViewController: topViewController)
//
//        // Set the window's root view controller to the navigation controller
//        window.rootViewController = navigationController
//
//        // Configure the window
//        window.backgroundColor = UIColor.systemBackground
//        window.makeKeyAndVisible()
//        
//        print("Override point for customization after application launch.")
//        
//        return true
//    }
//
//    // MARK: UISceneSession Lifecycle
//
//    func application(_ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession, options: UIScene.ConnectionOptions) -> UISceneConfiguration {
//        // This method won't be called since we removed UIScene configuration from Info.plist
//        fatalError("This method should not be called since UIScene configuration is removed.")
//    }
//
//    func application(_ application: UIApplication, didDiscardSceneSessions sceneSessions: Set<UISceneSession>) {
//        // This method won't be called since we removed UIScene configuration from Info.plist
//        fatalError("This method should not be called since UIScene configuration is removed.")
//    }
//}

